/*using System;
using System.Diagnostics;
using System.IO;
using OpenCvSharp;

class Program
{
    static Mat CreateGaussianKernel(int size, double sigma)
    {
        Mat kernel = new Mat(size, size, MatType.CV_64F);
        int center = size / 2;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                double diffX = x - center;
                double diffY = y - center;
                kernel.Set(y, x, Math.Exp(-(diffX * diffX + diffY * diffY) / (2 * sigma * sigma)));
            }
        }
        Cv2.Normalize(kernel, kernel, 1.0, 0, NormTypes.L1);
        return kernel;
    }

    static void Main()
    {
        string solutionDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName;
        string imagesDir = Path.Combine(solutionDir, "images");
        string inputImagePath = Path.Combine(imagesDir, "resized_image_128x128.jpg");
        string outputImagePath = Path.Combine(imagesDir, "filtered_image_128x128.jpg");
        int radius = 350;
        double sigma = radius / 3.0;
        int kernelSize = 2 * radius + 1;
        using Mat image = Cv2.ImRead(inputImagePath, ImreadModes.Color);
        if (image.Empty())
        {
            Console.WriteLine("ERROR: Image loading failed");
            return;
        }
        var processSw = Stopwatch.StartNew();
        Mat kernel = CreateGaussianKernel(kernelSize, sigma);
        Mat[] channels = image.Split();
        Mat[] filteredChannels = new Mat[channels.Length];
        for (int i = 0; i < channels.Length; i++)
        {
            filteredChannels[i] = new Mat();
            for (int j = 0; j < 25; j++)
            {
                Cv2.Filter2D(channels[i], filteredChannels[i], -1, kernel,
                        new Point(-1, -1), 0, BorderTypes.Reflect);
            }
            channels[i].Dispose();
        }
        Mat result = new Mat();
        Cv2.Merge(filteredChannels, result);
        processSw.Stop();
        Cv2.ImWrite(outputImagePath, result);
        Console.WriteLine($"Total Time: {processSw.Elapsed.TotalMilliseconds:0.00}ms");
    }
}*/