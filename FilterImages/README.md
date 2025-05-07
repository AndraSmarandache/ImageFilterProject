# Gaussian Blur Image Processor

A C# implementation of sequential Gaussian blur filtering, preparing for parallel versions (MPI/CUDA).

## What's Here
- Single executable that processes images in `images/` folder
- Applies 3×3 Gaussian blur with σ=30 intensity
- Measures and displays exact processing time
- Saves results as `filtered_[original_name].jpg`

## How to Run in Visual Studio

1. **Open Solution**:
   - Launch Visual Studio
   - Open `FilterImages.sln` from the root folder

2. **Run the Project**:
   - Press <kbd>F5</kbd> to debug, or
   - Use menu: `Debug > Start Without Debugging`

## Key Numbers
| Image Size | Blur Time |
|------------|----------|
| 128×128px  | 93ms     |
| 32K×32Kpx  | ~8.5min  |

Built with: .NET 9.0, SixLabors.ImageSharp

## Documentation
https://docs.google.com/document/d/1astXG8wdAVd-7KFACmJk3RbctEtsveufCPoktnsuwP8/edit?usp=sharing
