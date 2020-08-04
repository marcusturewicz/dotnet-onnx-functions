using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DotNetOnnxFunctions
{
    public class PredictFunction
    {
        [FunctionName("Predict")]
        public static IActionResult Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)]
            HttpRequest req,
            ExecutionContext context)
        {
            // Read image from request
            using var imageInStream = new MemoryStream();
            req.Body.CopyTo(imageInStream);
            using var image = Image.Load<Rgb24>(imageInStream.ToArray(), out IImageFormat format);

            // Resize image
            using var imageOutStream = new MemoryStream();
            image.Mutate(i =>
            {
                i.Resize(new ResizeOptions()
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });
            image.Save(imageOutStream, format);

            // Preprocess image and create input tensor
            var input = new DenseTensor<float>(new[] { 1, image.Height, image.Width, 3 });
            for (int y = 0; y < image.Height; y++)
            {
                var pixelRow = image.GetPixelRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = pixelRow[x];
                    input[0, y, x, 0] = (pixel.R - 127) / 128f;
                    input[0, y, x, 1] = (pixel.G - 127) / 128f;
                    input[0, y, x, 2] = (pixel.B - 127) / 128f;
                }
            }

            // Run inference
            var inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor("images:0", input)
            };
            var outputs = new List<string> { "Softmax:0" };
            using var session = new InferenceSession(Path.Combine(context.FunctionAppDirectory, "efficientnet-lite4.onnx"));
            using var results = session.Run(inputs, outputs);

            // Format the prediction
            var prediction = (results.First().Value as IEnumerable<float>)
                .Select((x, i) => new { Label = LabelMap.Labels[i], Confidence = x })
                .OrderByDescending(x => x.Confidence)
                .Take(10)
                .ToArray();

            return new OkObjectResult(prediction);
        }
    }
}
