## Neural Network Calibrations
This repo contains the Temperature Scaling (global and class-wise) [1] calibration methods for deep neural networks.

## Setup
- python 3.x
- pytorch>=1.8
- matplotlib
- numpy

## Temperature scaling
The temperature scaling [1] approaches are implemented within the `ModelWithTemperature` class. This class can either learn or grid-search for an appropriate temperature scalar both globally or per-class.

#### Example (Grid Search)
The following example finds temperature using grid search. The learned approach is commented out and can be swapped with grid search.
```python
# Setup values to iterate over during learning or grid search
# For the grid search approach
temps = torch.linspace(0.25, 4.0, 100)
# For the learned approach
lrs = torch.linspace(0.001, 0.01, 10)
num_iters = torch.linspace(50, 100, 10)

# load base model
base_model = load_model(model_path)
device = 'cpu'
# number of bins in the histogram
n_bins = 15

# Get per-class or global ece scores
# ece_criterion = PerClassECE(n_bins=NBINS)
ece_criterion = ECE(n_bins=NBINS)

# Compute ece of model on validation dataloader
before_ece = ece_criterion.compute_ece(model, val_loader)
print(f"Before ece: {before_ece}")
# Reliability diagrams and bin count diagram
ece_criterion.reliability_diagram_and_bin_count()


# Create new model with temperature scaling
# UNCOMMENT FOR LEARNED TEMPERATURE SCALING
# model_temp = ModelWithTemperature(model=base_model,
#                            strategy="learn", per_class=True,
#                            device=device, n_bins=NBINS)
# AND COMMENT OUT FOR LEARNED TEMPERATURE SCALING
model_temp = ModelWithTemperature(model=base_model,
                            strategy="grid", per_class=True,
                            device=device, n_bins=NBINS)


# Set temperature of model using grid search
# only t_vals required for grid
# only lrs and num_iters required for learn
# UNCOMMENT FOR LEARNED TEMPERATURE SCALING
# temperature = model_temp.set_temperature(val_loader,
#                            lrs=list(lrs),
#                            num_iters=list(num_iters))
# AND COMMENT OUT FOR LEARNED TEMPERATURE SCALING
temperature = model_temp.set_temperature(val_loader,
                            t_vals=list(temps))


# Compute ece of temperature-scaled model on validation set
after_ece = ece_criterion.compute_ece(model_temp, val_loader)
# Reliability diagrams and bin count diagram
ece_criterion.reliability_diagram_and_bin_count()

print(f"Final temperature: {temperature}")
print(f"After ece: {after_ece}")
```

![reliability_diagram](../assets/reliability_diagram.png)

## Reference
[1] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks," In ICML, pp. 2130–2143, 2017.
Available: https://arxiv.org/abs/1706.04599v2.
