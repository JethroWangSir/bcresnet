# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import wandb
from thop import profile
import json

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset
from loss import softmax_loss, weighted_focal_loss


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """

        parser = ArgumentParser()
        parser.add_argument("--ver", default=1, help="google speech command set version 1 or 2", type=int)
        parser.add_argument("--num_classes", default=12, help="number of classes", type=int)
        parser.add_argument("--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8])
        parser.add_argument("--lambda1", default=1, help="weight of keyword branch", type=float)
        parser.add_argument("--lambda2", default=1, help="weight of speech branch", type=float)
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        parser.add_argument("--eval", help="Only run evaluation", action="store_true")
        parser.add_argument("--ckpt", help="Path to checkpoint file for evaluation", type=str, default="")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

        # Add a list to track top 3 validation accuracies
        self.top_3_valid_accs = []
        
        # Create a directory to save checkpoints if it doesn't exist
        self.checkpoint_dir = f"./checkpoints/sr_tau_{self.tau}_ver_{self.ver}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.eval and not self.ckpt:
            raise ValueError("Please provide a checkpoint file using --ckpt <path> when using --eval mode.")

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """

        wandb.init(entity="jethrowang0531", project="BC-ResNet", name=f'sr_tau_{self.tau}_ver_{self.ver}')

        # train hyperparameters
        total_epoch = 200
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer_encoder = torch.optim.SGD(
            list(self.model.cnn_head.parameters()) + list(self.model.BCBlocks.parameters()), 
            lr=0, weight_decay=1e-3, momentum=0.9
        )
        optimizer_cls1 = torch.optim.SGD(self.model.classifier1.parameters(), lr=0, momentum=0.9)
        optimizer_cls2 = torch.optim.SGD(self.model.classifier2.parameters(), lr=0, momentum=0.9)
        optimizer_cls3 = torch.optim.SGD(self.model.classifier3.parameters(), lr=0, momentum=0.9)
        
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # Best model tracking
        best_valid_acc = 0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer_encoder.param_groups:
                    param_group["lr"] = lr
                for param_group in optimizer_cls1.param_groups:
                    param_group["lr"] = lr
                for param_group in optimizer_cls2.param_groups:
                    param_group["lr"] = lr
                for param_group in optimizer_cls3.param_groups:
                    param_group["lr"] = lr

                # Extract inputs and labels
                inputs, labels = sample
                inputs = inputs.to(self.device)
                print(f'inputs: {inputs.shape}')
                labels = labels.to(self.device)
                print(f'labels: {labels.shape}, {labels}')

                # Define labels1, labels2, labels3
                labels1 = (labels != 0).long()  # 0 -> non-speech, 1~11 -> speech
                print(f'labels1: {labels1.shape}, {labels1}')
                labels2 = labels[labels > 0]
                labels2 = (labels2 >= 2).long()   # 1 -> non-keyword, 2~11 -> keyword
                print(f'labels2: {labels2.shape}, {labels2}')
                labels3 = labels[labels > 1]
                labels3 = torch.where(labels3 >= 2, labels3 - 2, torch.tensor(-1, device=self.device))  # labels3 keeps only 2~11 (mapped to 0~9), others are set to -1 (invalid labels)
                print(f'labels3: {labels3.shape}, {labels3}')

                # Preprocess inputs
                inputs = self.preprocess_train(inputs, labels, augment=True)
                print(f'processed_inputs: {inputs.shape}')

                # Get embeddings
                embeddings = self.model.encode(inputs)
                print(f'embeddings: {embeddings.shape}')
                
                # Classify for speech/non-speech
                outputs1 = self.model.speech_branch(embeddings)  # Speech/Non-speech
                print(f'outputs1: {outputs1.shape}')

                # Only pass embeddings with labels 1–11 to keyword_branch
                keyword_embeddings = embeddings[labels > 0]
                outputs2 = self.model.keyword_branch(keyword_embeddings)  # Keyword/Non-keyword
                print(f'outputs2: {outputs2.shape}')

                # Only pass embeddings with labels 2–11 to keyword_classification
                keyword_class_embeddings = embeddings[labels >= 2]
                outputs3 = self.model.keyword_classification(keyword_class_embeddings)  # Keyword classification (10 classes)
                print(f'outputs3: {outputs3.shape}')

                # Compute Losses
                loss_speech = weighted_focal_loss(outputs1, labels1)
                loss_keyword = weighted_focal_loss(outputs2, labels2)  # Only compute for 1~11
                loss_softmax = softmax_loss(outputs3, labels3)  # Only compute for 2~11
                loss = loss_softmax + self.lambda1 * loss_keyword + self.lambda2 * loss_speech
                wandb.log({"Total Loss": loss.item(), "Softmax Loss": loss_softmax.item(), "Keyword Loss": loss_keyword.item(), "Speech Loss": loss_speech.item()})

                loss.backward()

                # Update the encoder and classifiers
                optimizer_encoder.step()
                optimizer_cls1.step()
                optimizer_cls2.step()
                optimizer_cls3.step()

                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            wandb.log({"LR": lr})
            with torch.no_grad():
                self.model.eval()
                valid_acc, valid_auroc, valid_f1, valid_fa = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print(f"Valid - Acc: {valid_acc:.3f}, AUROC: {valid_auroc:.3f}, F1: {valid_f1:.3f}, FA: {valid_fa:.3f}")
                wandb.log({
                    "Epoch": epoch + 1,
                    "Valid_Acc": valid_acc,
                    "Valid_AUROC": valid_auroc,
                    "Valid_F1": valid_f1,
                    "Valid_FA": valid_fa
                })

                # Save checkpoint for top 3 validation accuracies
                self._save_top_3_checkpoints(epoch, valid_acc)

        test_acc, test_auroc, test_f1, test_fa = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset
        print(f"Last ckpt test - Acc: {test_acc:.3f}, AUROC: {test_auroc:.3f}, F1: {test_f1:.3f}, FA: {test_fa:.3f}")

        # After training, test the best checkpoint
        self._test_best_checkpoint()

        wandb.finish()

        print("End.")

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset and calculates accuracy, AUROC, F1-score, and false alarm rate.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
            float: The AUROC score for the multi-class classification task.
            float: The F1-score for the multi-class classification task.
            float: The false alarm rate (FA), where label 0 or 1 is misclassified as label 2~11.
        """

        self.model.eval()

        all_labels = []
        all_outputs = []  # probabilities
        all_predictions = []

        true_count = 0.0
        num_testdata = float(len(dataset))
        fa_count = 0
        neg_total = 0
        confusion_mat = np.zeros((self.num_classes, self.num_classes))

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)  # already probabilities

            # Collect all predictions and labels
            prediction = torch.argmax(outputs, dim=-1)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())  # probabilities
            all_predictions.extend(prediction.cpu().numpy())

            # Update confusion matrix
            batch_confusion = confusion_matrix(labels.cpu().numpy(), prediction.cpu().numpy(), labels=np.arange(self.num_classes))
            confusion_mat += batch_confusion            

            # Accuracy calculation
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage

        # AUROC calculation
        if len(set(all_labels)) < self.num_classes:
            auroc = float('nan')
        else:
            auroc = roc_auc_score(np.array(all_labels), np.array(all_outputs), average='macro', multi_class='ovr') * 100.0
        
        # F1-score calculation
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='macro') * 100.0
        
        # False alarm rate calculation
        for i in [0, 1]:  # Only consider label 0 (_silence_) and label 1 (_unknown_)
            fa_count += np.sum(confusion_mat[i, 2:])  # Count misclassifications to 2~11
            neg_total += np.sum(confusion_mat[i, :])   # Total occurrences of class 0 or 1
        if neg_total == 0:
            fa = None
        else:
            fa = fa_count / neg_total * 100.0

        return acc, auroc, f1, fa

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """

        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("/share/nas169/jethrowang/DB/GSC"):
            os.mkdir("/share/nas169/jethrowang/DB/GSC")
        base_dir = "/share/nas169/jethrowang/DB/GSC/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        test_dir = base_dir.replace("commands", "commands_test_set")
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)
            print("Done...")

        # Define data loaders
        train_dir = "%s/train_12class" % base_dir
        valid_dir = "%s/valid_12class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        transform = transforms.Compose([Padding()])
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f+SR on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

    def _save_top_3_checkpoints(self, epoch, valid_acc):
        """
        Save checkpoints for top 3 validation accuracies.
        
        Parameters:
            epoch (int): Current training epoch
            valid_acc (float): Validation accuracy for the current epoch
        """

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'valid_acc': valid_acc
        }

        # If less than 3 best accuracies, always save
        if len(self.top_3_valid_accs) < 3:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}_acc_{valid_acc:.2f}.ckpt')
            torch.save(checkpoint, checkpoint_path)
            self.top_3_valid_accs.append((valid_acc, checkpoint_path))
            self.top_3_valid_accs.sort(reverse=True)  # Sort in descending order
        else:
            # Check if current accuracy is better than the worst in top 3
            if valid_acc > self.top_3_valid_accs[-1][0]:
                # Remove the worst checkpoint
                _, worst_path = self.top_3_valid_accs.pop()
                os.remove(worst_path)

                # Save new checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}_acc_{valid_acc:.2f}.ckpt')
                torch.save(checkpoint, checkpoint_path)
                self.top_3_valid_accs.append((valid_acc, checkpoint_path))
                self.top_3_valid_accs.sort(reverse=True)  # Sort in descending order

        # Log the current top 3 checkpoint paths
        print("Current top 3 validation accuracy checkpoints:")
        for acc, path in self.top_3_valid_accs:
            print(f"Acc: {acc:.3f}, Path: {path}")
    
    def _calculate_params(self, model):
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, trainable_params

    def _calculate_macs(self, model):
        # Calculate MACs (Multiply-Accumulate Operations)
        input_sample = torch.randn(1, 1, 40, 87).to(self.device)
        macs, _ = profile(model, inputs=(input_sample,), verbose=False)

        return macs

    def _test_best_checkpoint(self):
        """
        Load and test the best checkpoint from the top 3 validation accuracies.
        """
        if not self.top_3_valid_accs:
            print("No checkpoints were saved. Skipping best checkpoint test.")
            return

        # Sort checkpoints by validation accuracy in descending order
        sorted_checkpoints = sorted(self.top_3_valid_accs, reverse=True)
        
        # Select the best checkpoint
        best_valid_acc, best_checkpoint_path = sorted_checkpoints[0]
        
        print(f"\nTesting best checkpoint with validation accuracy: {best_valid_acc:.3f}")
        print(f"Checkpoint path: {best_checkpoint_path}")

        # Load the best checkpoint
        checkpoint = torch.load(best_checkpoint_path)
        
        # Create a new model instance and load the state dict
        best_model = BCResNets(int(self.tau * 8)).to(self.device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        best_model.eval()

        # Replace the current model with the best model for testing
        original_model = self.model
        self.model = best_model

        # Run test on the loaded model
        with torch.no_grad():
            best_test_acc, best_test_auroc, best_test_f1, best_test_fa = self.Test(self.test_dataset, self.test_loader, augment=False)
            print(f"Best ckpt test - Acc: {best_test_acc:.3f}, AUROC: {best_test_auroc:.3f}, F1: {best_test_f1:.3f}, FA: {best_test_fa:.3f}")
        
        # Calculate number of parameters
        total_params, trainable_params = self._calculate_params(self.model)

        # Calculate MACs (Multiply-Accumulate Operations)
        macs = self._calculate_macs(self.model)

        # Prepare results dictionary
        results = {
            'accuracy': best_test_acc,
            'auroc': best_test_auroc,
            'f1-score': best_test_f1,
            'false_alarm': best_test_fa,
            'params': {
                'total_params_k': total_params/1000,
                'trainable_params_k': trainable_params/1000
            },
            'macs_m': macs/1e6
        }

        # Save results to JSON
        results_path = os.path.join(self.checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Restore the original model
        self.model = original_model
    
    def Evaluation(self):
        print(f'Loading model: {self.ckpt}')
        eval_ckpt = torch.load(self.ckpt)
        self.model.load_state_dict(eval_ckpt['model_state_dict'])

        # Calculate number of parameters
        total_params, trainable_params = self._calculate_params(self.model)

        # Calculate MACs (Multiply-Accumulate Operations)
        macs = self._calculate_macs(self.model)

        # Perform evaluation
        with torch.no_grad():
            eval_acc, eval_auroc, eval_f1, eval_fa = self.Test(self.test_dataset, self.test_loader, augment=False)
            
        # Print results
        print(f"Eval - Acc: {eval_acc:.3f}, AUROC: {eval_auroc:.3f}, F1: {eval_f1:.3f}, FA: {eval_fa:.3f}")
        print(f"Params - Total: {total_params/1000:.2f}k, Trainable: {trainable_params/1000:.2f}k")
        print(f"MACs: {macs/1e6:.2f}M")

        # Prepare results dictionary
        results = {
            'accuracy': eval_acc,
            'auroc': eval_auroc,
            'f1-score': eval_f1,
            'false_alarm': eval_fa,
            'params': {
                'total_params_k': total_params/1000,
                'trainable_params_k': trainable_params/1000
            },
            'macs_m': macs/1e6
        }

        # Save results to JSON in the same directory as the checkpoint
        results_path = os.path.join(os.path.dirname(self.ckpt), 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    _trainer = Trainer()
    if _trainer.eval:
        _trainer.Evaluation()
    else:
        _trainer()
