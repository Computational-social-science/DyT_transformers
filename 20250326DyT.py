import torch
import torch.nn as nn
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# DyT Layer Implementation
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

# Vanilla Transformer Implementation
class VanillaTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )  
    def forward(self, src, tgt):
        return self.transformer(src, tgt)

# DyT Transformer Implementation
class DyTTransformer(nn.Module):
   def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            norm_first=False,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            norm_first=False,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.dyt = DyT(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers) 
   def forward(self, src, tgt):
        src = self.dyt(src)
        tgt = self.dyt(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

# Dummy Dataset for Testing
class DummyDataset(Dataset):
    def __init__(self, size=10, seq_len=20, d_model=512): #调试时控制size在10以内
        self.size = size
        self.seq_len = seq_len
        self.data = torch.randn(size, seq_len, d_model)
        self.target = torch.randn(size, seq_len, d_model)   
    def __len__(self):
        return len(self.data) 
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()  
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()   
    end_time = time.time()
    return total_loss / len(dataloader), end_time - start_time

def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    """Train model for multiple epochs"""
    print(f"Starting training for {model.__class__.__name__}...")
    times = []
    losses = []    
    for epoch in range(num_epochs):
        loss, epoch_time = train_epoch(model, dataloader, criterion, optimizer, device)
        times.append(epoch_time)
        losses.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")    
    return times, losses

def calculate_parameters(model):
    """Calculate number of parameters in model"""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, test_data, test_target, device, warmup_rounds=5):
    """Measure inference time for model"""
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_rounds):
            _ = model(test_data, test_target)        
        # Measure time
        start_time = time.time()
        _ = model(test_data, test_target)
        inference_time = time.time() - start_time        
        return inference_time

def get_activation_stats(model, data, target):
    """Get activation statistics for DyT layer"""
    activations = []    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())    
    if isinstance(model, DyTTransformer):
        hook = model.dyt.register_forward_hook(hook_fn)
        _ = model(data, target)
        hook.remove()        
        if len(activations) > 0:
            act_mean = np.mean(activations[0])
            act_std = np.std(activations[0])
            act_min = np.min(activations[0])
            act_max = np.max(activations[0])            
            return act_mean, act_std, act_min, act_max, activations[0]
    return None, None, None, None, None

def collect_activation_samples(model, device, d_model, num_samples=10):
    """Collect activation samples for visualization"""
    activation_samples = []    
    for _ in range(num_samples):
        sample_data = torch.randn(1, 20, d_model).to(device)
        sample_target = torch.randn(1, 20, d_model).to(device)
        _, _, _, _, act_data = get_activation_stats(model, sample_data, sample_target)
        if act_data is not None:
            activation_samples.append(act_data.flatten())    
    return activation_samples

def plot_training_time_comparison(epochs, vanilla_times, dyt_times, ax=None):
    """Plot training time comparison"""
    if ax is None:
        ax = plt.subplot(2, 3, 1)    
    ax.plot(epochs, vanilla_times, 'b-o', label='Transformer', markersize=4, linewidth=1.2)
    ax.plot(epochs, dyt_times, 'r-o', label='Transformer (DyT)', markersize=4, linewidth=1.2)
    ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    # 修改图例样式，添加边框
    ax.legend(frameon=True, fontsize=10, 
             facecolor='white', edgecolor='gray', framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add subplot label aligned with title
    ax.text(-0.1,1.0, 'A', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def plot_inference_time_comparison(vanilla_inference_time, dyt_inference_time, ax=None):
    """Plot inference time comparison"""
    if ax is None:
        ax = plt.subplot(2, 3, 2)    
    # Calculate speedup ratio
    speedup = vanilla_inference_time / dyt_inference_time    
    models = ['Transformer', 'Transformer (DyT)']
    # Normalize inference times for better visualization
    norm_times = [1.0, dyt_inference_time/vanilla_inference_time]    
    bars = ax.bar(models, norm_times, color=['#4472C4', '#ED7D31'], width=0.6, 
                 edgecolor='black', linewidth=0.5)
    ax.set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Time', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        time_val = vanilla_inference_time if i == 0 else dyt_inference_time
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)    
    # Add speedup text as x-axis label
    ax.set_xlabel(f'Speedup: {speedup:.2f}x', fontsize=12)    
    # Set y-axis limit to avoid overlap
    ax.set_ylim(0, max(norm_times) * 1.2)    
    # Add subplot label aligned with title
    ax.text(-0.1,1.0, 'B', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def plot_loss_comparison(epochs, vanilla_losses, dyt_losses, ax=None):
    """Plot loss comparison"""
    if ax is None:
        ax = plt.subplot(2, 3, 3)    
    ax.plot(epochs, vanilla_losses, 'b-o', label='Transformer', markersize=4, linewidth=1.2)
    ax.plot(epochs, dyt_losses, 'r-o', label='Transformer (DyT)', markersize=4, linewidth=1.2)
    ax.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    # 修改图例样式，添加边框
    ax.legend(frameon=True, fontsize=10,
             facecolor='white', edgecolor='gray', framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)    
    # Add subplot label aligned with title
    ax.text(-0.1, 1.0, 'C', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def plot_parameter_comparison(vanilla_params, dyt_params, ax=None):
    """Plot parameter comparison"""
    if ax is None:
        ax = plt.subplot(2, 3, 4)    
    models = ['Transformer', 'Transformer (DyT)']
    # Convert to millions for better readability on y-axis but display actual values
    params_in_millions = [p/1e6 for p in [vanilla_params, dyt_params]]
    params_actual = [vanilla_params, dyt_params]
    colors = ['#4472C4', '#ED7D31']    
    # Create bar chart
    bars = ax.bar(models, params_in_millions, color=colors, width=0.6, 
                 edgecolor='black', linewidth=0.5)    
    # Add actual parameter values to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{params_actual[i]:,}', ha='center', va='bottom', fontsize=9)    
    ax.set_title('Parameter Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Parameters (millions)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)    
    # Calculate parameter optimization
    param_reduction = vanilla_params - dyt_params
    optimization_percentage = (param_reduction / vanilla_params) * 100    
    # Add parameter optimization as x-axis label
    if param_reduction > 0:
        ax.set_xlabel(f'Parameter Optimization: {param_reduction:,}',fontsize=12)
    else:
        ax.set_xlabel(f'Parameter Increase: {-param_reduction:,}',fontsize=12)
    
    # Set y-axis limit to avoid overlap
    ax.set_ylim(0, max(params_in_millions) * 1.15)    
    # Add subplot label aligned with title
    ax.text(-0.1, 1.0, 'D', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def plot_activation_distribution(activation_samples, act_mean, act_std, act_min, act_max, ax=None):
    """Plot activation distribution using boxplot visualization"""
    if ax is None:
        ax = plt.subplot(2, 3, 5)    
    # Use original activation samples without normalization
    box = ax.boxplot(activation_samples, patch_artist=True)    
    # Customize boxplot colors
    for patch in box['boxes']:
        patch.set_facecolor('#ED7D31')
        patch.set_alpha(0.7)    
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=1.2, linestyle='-')    
    for cap in box['caps']:
        cap.set(color='black', linewidth=1.2)    
    for median in box['medians']:
        median.set(color='blue', linewidth=1.2)    
    for flier in box['fliers']:
        flier.set(marker='o', markerfacecolor='red', markersize=3, alpha=0.5)
    
    # Set titles and labels with adjusted font sizes
    ax.set_title('Activation Distribution (DyT)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Restore x-tick labels with sample indices
    ax.set_xticklabels([str(i+1) for i in range(len(activation_samples))])    
    # Set y-axis limit to leave space for legend
    y_max = max([max(sample) for sample in activation_samples]) if activation_samples else act_max
    y_min = min([min(sample) for sample in activation_samples]) if activation_samples else act_min
    y_range = y_max - y_min    
    # Add extra space at the top for the legend and adjust bottom for better visualization
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.7)    
    # Create a comprehensive legend explaining boxplot elements
    combined_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Median'),
        plt.Rectangle((0,0), 1, 1, fc='#ED7D31', alpha=0.7, label='IQR (25-75%)'),
        plt.Line2D([0], [0], color='black', linewidth=1.2, label='Range'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, alpha=0.5, label='Outliers'),
    ]   
    # Create a single legend with boxplot elements
    ax.legend(handles=combined_elements, loc='upper right', 
             frameon=True, fontsize=10, 
             bbox_to_anchor=(0.98, 0.98), 
             facecolor='white', edgecolor='gray', framealpha=0.8)    
    # Add subplot label aligned with title
    ax.text(-0.1, 1.0, 'E', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def plot_activation_function_comparison(alpha_value, ax=None):
    """Plot activation function comparison with multiple alpha values"""
    if ax is None:
        ax = plt.subplot(2, 3, 6)    
    # Extended input range from -6 to 6
    x = np.linspace(-6, 6, 1000)    
    # Calculate different activation curves
    y_standard = np.tanh(x)
    y_dyt = np.tanh(alpha_value * x)
    y_steep = np.tanh(4 * x)
    y_gentle = np.tanh(0.25 * x)
    
    # Plot with consistent styling and distinct colors
    ax.plot(x, y_standard, 'k-', label='tanh(x)', linewidth=1.5)
    ax.plot(x, y_dyt, 'r-', label=f'tanh({alpha_value:.2f}x)', linewidth=1.5)
    ax.plot(x, y_steep, 'b--', label='tanh(4x)', linewidth=1.5)
    ax.plot(x, y_gentle, 'g--', label='tanh(x/4)', linewidth=1.5)    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)    
    # Set titles and labels with consistent font sizes
    ax.set_title('Activation Function Curves', fontsize=13, fontweight='bold')
    ax.set_xlabel('Input', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)    
    # Set axis limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1.1, 1.1)    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)    
    # 将图例从右中间移到右下角
    ax.legend(loc='lower right', frameon=True, fontsize=10,
             bbox_to_anchor=(0.98, 0.02),
             facecolor='white', edgecolor='gray', framealpha=0.8)    
    # Add subplot label
    ax.text(-0.1, 1.0, 'F', transform=ax.transAxes, fontsize=15, 
            fontweight='bold', va='center', ha='right')

def save_plots(formats=None):
    """Save plots in multiple formats"""
    if formats is None:
        formats = ['png', 'jpg', 'tiff', 'svg']    
    for fmt in formats:
        filename = f'transformer_comparison.{fmt}'
        plt.savefig(filename, dpi=400, format=fmt, bbox_inches='tight')
        print(f"Plot saved as {filename}")

def main():
    # Setup parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 512
    nhead = 8
    num_layers = 6
    batch_size = 32
    num_epochs = 300 #调试过程中尽量控制在10以内
 
    # Prepare data
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
    # Initialize models
    vanilla_transformer = VanillaTransformer(d_model, nhead, num_layers).to(device)
    dyt_transformer = DyTTransformer(d_model, nhead, num_layers).to(device)
   
    # Define loss function and optimizers
    criterion = nn.MSELoss()
    vanilla_optimizer = torch.optim.Adam(vanilla_transformer.parameters())
    dyt_optimizer = torch.optim.Adam(dyt_transformer.parameters())
   
    # Train models
    vanilla_times, vanilla_losses = train_model(vanilla_transformer, dataloader, criterion, vanilla_optimizer, device, num_epochs)
    dyt_times, dyt_losses = train_model(dyt_transformer, dataloader, criterion, dyt_optimizer, device, num_epochs)
   
    # Output performance comparison
    print("\nPerformance Comparison:")
    print(f"Vanilla Transformer average training time per epoch: {sum(vanilla_times)/len(vanilla_times):.2f}s")
    print(f"DyT Transformer average training time per epoch: {sum(dyt_times)/len(dyt_times):.2f}s")
    print(f"Vanilla Transformer final loss: {vanilla_losses[-1]:.4f}")
    print(f"DyT Transformer final loss: {dyt_losses[-1]:.4f}")

    # Calculate parameters
    vanilla_params = calculate_parameters(vanilla_transformer)
    dyt_params = calculate_parameters(dyt_transformer)
    print(f"\nParameter Comparison:")
    print(f"Vanilla Transformer parameters: {vanilla_params:,}")
    print(f"DyT Transformer parameters: {dyt_params:,}")
    
    # Measure inference time
    print("\nInference Time Comparison:")
    test_data = torch.randn(10, 20, d_model).to(device)
    test_target = torch.randn(10, 20, d_model).to(device)
    
    vanilla_inference_time = measure_inference_time(vanilla_transformer, test_data, test_target, device)
    dyt_inference_time = measure_inference_time(dyt_transformer, test_data, test_target, device)
    
    print(f"Vanilla Transformer inference time: {vanilla_inference_time:.4f} seconds")
    print(f"DyT Transformer inference time: {dyt_inference_time:.4f} seconds")
    
    # Get DyT activation statistics
    sample_data = next(iter(dataloader))[0][:1].to(device)
    sample_target = next(iter(dataloader))[1][:1].to(device)
    act_mean, act_std, act_min, act_max, first_activation = get_activation_stats(dyt_transformer, sample_data, sample_target)
    
    if act_mean is not None:
        print("\nDyT Activation Statistics:")
        print(f"Mean: {act_mean:.4f}")
        print(f"Standard Deviation: {act_std:.4f}")
        print(f"Minimum: {act_min:.4f}")
        print(f"Maximum: {act_max:.4f}")
    
    # Collect activation samples
    activation_samples = collect_activation_samples(dyt_transformer, device, d_model)

    # Create visualization with optimized layout
    plt.figure(figsize=(16, 9), facecolor='white')
    
   # Set publication-quality parameters with larger font sizes
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,                # Increased from 10
        'axes.linewidth': 1.0,
        'axes.labelsize': 12,           # Increased from 11
        'axes.titlesize': 13,           # Increased from 12
        'xtick.labelsize': 10,          # Increased from 9
        'ytick.labelsize': 10,          # Increased from 9
        'legend.fontsize': 10,          # Increased from 9
        'figure.titlesize': 15          # Increased from 14
    })

    # Create custom grid layout with increased spacing for better readability
    gs = plt.GridSpec(2, 3, figure=plt.gcf(), 
                    width_ratios=[1, 1, 1], 
                    height_ratios=[1, 1],
                    wspace=0.25, hspace=0.25)  # Increased spacing to avoid overlap
    
    # Create axes with the grid layout
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])
    
    # Define epochs range for plotting
    epochs = range(1, num_epochs + 1)
    
    # Plot with specific axes
    plot_training_time_comparison(epochs, vanilla_times, dyt_times, ax=ax1)
    plot_inference_time_comparison(vanilla_inference_time, dyt_inference_time, ax=ax2)
    plot_loss_comparison(epochs, vanilla_losses, dyt_losses, ax=ax3)
    plot_parameter_comparison(vanilla_params, dyt_params, ax=ax4)
    
    if act_mean is not None and activation_samples:
        plot_activation_distribution(activation_samples, act_mean, act_std, act_min, act_max, ax=ax5)
        
        # Get alpha parameter value
        alpha_value = dyt_transformer.dyt.alpha.item()
        plot_activation_function_comparison(alpha_value, ax=ax6)
    
    # Add a main title with adjusted position and increased font size
    plt.suptitle('Performance Comparison: Transformer (Layer Norm) vs Transformer (DyT)', 
                fontsize=15, fontweight='bold', y=0.99)

    # Use figure-level adjustment with optimized margins to prevent overlap
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.09)

    # Save plots in multiple formats with high resolution
    save_plots(['png', 'jpg', 'tiff', 'svg'])
    plt.show()

if __name__ == "__main__":
    main()