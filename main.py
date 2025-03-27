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

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--num_classes", default=12, help="number of classes", type=int
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = 200
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

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
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                # wandb.log({"Softmax Loss": loss.item()})
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            # wandb.log({"LR": lr.item()})
            with torch.no_grad():
                self.model.eval()
                valid_acc, valid_auroc, valid_f1, valid_fa = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid - acc: %.3f, auroc: %.3f, f1: %.3f, fa: %.3f" % (valid_acc, valid_auroc, valid_f1, valid_fa))
                # wandb.log({
                #     "Valid Acc": valid_acc,
                #     "Valid AUROC": valid_auroc,
                #     "Valid F1": valid_f1,
                #     "Valid FA": valid_fa
                # })

        test_acc, test_auroc, test_f1, test_fa = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset
        print("test - acc: %.3f, auroc: %.3f, f1: %.3f, fa: %.3f" % (test_acc, test_auroc, test_f1, test_fa))
        # wandb.log({
        #     "Epoch": epoch,
        #     "Test Acc": test_acc,
        #     "Test AUROC": test_auroc,
        #     "Test F1": test_f1,
        #     "Test FA": test_fa
        # })
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
            float: The false alarm rate (FA).
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        all_labels = []
        all_outputs = []  # logits
        all_predictions = []
        confusion_mat = np.zeros((self.num_classes, self.num_classes))

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)

            # Collect all predictions and labels
            prediction = torch.argmax(outputs, dim=-1)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())  # logits
            all_predictions.extend(prediction.cpu().numpy())

            # Update confusion matrix
            batch_confusion = confusion_matrix(labels.cpu().numpy(), prediction.cpu().numpy(), labels=np.arange(self.num_classes))
            confusion_mat += batch_confusion            

            # Accuracy calculation
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage

        # AUROC calculation
        all_outputs_prob = torch.softmax(torch.tensor(all_outputs), dim=1).cpu().detach().numpy()
        auroc = roc_auc_score(np.array(all_labels), all_outputs_prob, average='macro', multi_class='ovr')
        
        # F1-score calculation
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='macro')
        
        # False alarm rate calculation
        fa = {}
        for i in range(self.num_classes):
            false_alarms = np.sum(confusion_mat[:, i]) - confusion_mat[i, i]
            total_predictions = np.sum(confusion_mat[:, i]) + confusion_mat[i, i]
            fa[i] = false_alarms / total_predictions * 100 if total_predictions != 0 else 0

        return acc, auroc, f1, fa

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("/share/nas169/jethrowang/GSC"):
            os.mkdir("/share/nas169/jethrowang/GSC")
        base_dir = "/share/nas169/jethrowang/GSC/speech_commands_v0.01"
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
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

    def save_checkpoint(self, filepath='model.ckpt'):
        """
        Save the current model checkpoint to a file.

        Parameters:
            filepath (str): Path to save the checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add any other information you want to save, e.g., epoch, loss, etc.
        }
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")


if __name__ == "__main__":
    # wandb.init(project="BC-ResNet", name='BC-ResNet')

    _trainer = Trainer()
    _trainer()

    # wandb.finish()
