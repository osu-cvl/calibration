import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

class ModelWithTemperaturePerClass(nn.Module):
    """
    A thin decorator, which wraps a model with per-class temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, num_classes, n_bins=15, device='cpu'):
        super(ModelWithTemperaturePerClass, self).__init__()
        self.model = model.to(device)
        self.temperatures = None
        self.num_classes = num_classes
        self.temp_list = []
        self.device = device
        self.n_bins = n_bins

    def forward(self, input):
        input = input.to(self.device)
        logits = self.model(input)
        return self.temperature_scale(logits)

    def same_class_temperature_scale(self, logits):
        """
        Perform temperature scaling on logits using only one temperature
        """
        # Expand temperature to match the size of logits
        return logits / self.class_temperature.expand(logits.size(0), logits.size(1))

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Confirm temperatures have been set
        if self.temperatures is None:
            print("Set temperatures first!")
            return None

        # get argmax predictions
        preds = torch.argmax(logits, dim=1)

        # Get appropriate temperature values
        temperatures = self.temperatures[preds].squeeze(1)
        # Divide logits by appropriate temperatures
        return logits / temperatures.expand(logits.size(0), logits.size(1))

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.same_class_temperature_scale(class_logits), class_labels)
            loss.backward()
            return loss

        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = PerClassECE(self.num_classes, n_bins=self.n_bins)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels)
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))
        print('Before temperature - Mean Per-class ECE: %.3f' % (before_temperature_ece.mean()))

        # Iterate through classes and learn temperatures
        for i in range(self.num_classes):
            self.class_temperature = nn.Parameter(torch.ones((1,1), device=self.device) * 1.0)
            optimizer = optim.LBFGS([self.class_temperature], lr=0.01, max_iter=50)
            class_idx = torch.where(labels == i)[0]
            class_logits = logits[class_idx]
            class_labels = labels[class_idx]

            # Next: optimize the temperature w.r.t. NLL
            optimizer.step(eval)
            #print(f'Optimal temperature for Class {i}: {self.temperature.data.numpy():.3f}')
            self.temp_list.append(self.class_temperature.cpu().data.numpy())

        self.temperatures = torch.tensor(self.temp_list).to(self.device)


        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels)

        print(f'Optimal temperatures: {self.temperatures.cpu().squeeze().data.numpy()}')
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        print('After temperature - Mean Per-class ECE: %.3f' % (after_temperature_ece.mean()))

        return self.temperatures.squeeze().squeeze()

    
class PerClassECE(nn.Module):
    """
    Calculates the MEAN Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    """
    def __init__(self, num_classes, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(PerClassECE, self).__init__()
        self.num_classes = num_classes
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins

    def forward(self, logits, labels, sm=True):
        if not sm:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(self.num_classes, device=logits.device)
        self.bin_accuracy = torch.zeros((self.num_classes,self.n_bins))
        self.prop_in_bin = torch.zeros((self.num_classes,self.n_bins))
        self.count_in_bin = torch.zeros((self.num_classes,self.n_bins))

        for c in range(self.num_classes):
            class_idx = torch.where(predictions == c)[0]
            class_confidences = confidences[class_idx]
            class_accuracies = accuracies[class_idx]


            for i,(bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
                # Calculated |confidence - accuracy| in each bin
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                self.count_in_bin[c,i] =  in_bin.sum()
                self.prop_in_bin[c,i] = in_bin.float().mean()
                if self.prop_in_bin[c,i].item() > 0:
                    self.bin_accuracy[c,i] = class_accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    ece[c] += torch.abs(avg_confidence_in_bin - self.bin_accuracy[c,i]) * self.prop_in_bin[c,i]

        return ece

    def reliability_diagram_and_bin_count(self, title=""):
        fig_size = 12
        fig = plt.figure(figsize=(fig_size,fig_size//2))
        bin_width = 1./self.n_bins
        bin_center = torch.linspace(0.0+0.5*bin_width,1.0+0.5*bin_width,self.n_bins+1)[:-1]

        for c_idx in range(self.num_classes):
            ax0 = fig.add_subplot(self.num_classes,2,1+c_idx)
            ax0.title.set_text(title + f" Class-{c_idx}")
            ax0.bar(bin_center,self.bin_accuracy[c_idx],align='center',width=bin_width*0.7,label='bin precision',color='orange')
            ax0.set_xlim(0,1)
            ax0.set_ylim(0,1)
            ax0.plot(bin_center,bin_center,label='ideal case',color='blue',linestyle='-.')
            ax0.set_xlabel('estimated label posterior')
            ax0.set_ylabel('Actual accuracy')
            ax0.legend()

            ax1 = fig.add_subplot(self.num_classes,2,2+c_idx)
            ax1.bar(bin_center,self.count_in_bin[c_idx],align='center',width=bin_width*0.7,label='bin counts',color='blue')
            for k,c in enumerate(self.count_in_bin[c_idx]):
                    ax1.text(bin_center[k]-0.05,self.count_in_bin[c_idx,k]+100,str(int(c)),color='black',fontsize='small',fontweight='bold')

            ax1.set_xlim(0,1)
            ax1.set_xlabel('estimated label posterior')
            ax1.set_ylabel('example counts in bin')
            ax1.legend()

        fig.show()
        return fig
