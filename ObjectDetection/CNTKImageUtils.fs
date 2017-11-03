module CNTKImageUtils
open FSharp.Collections.ParallelSeq
open System.Drawing
open System.Drawing.Imaging
open System.Drawing.Drawing2D
open System.Runtime.InteropServices
open CNTK
open System.Threading.Tasks
open FSharp.NativeInterop

let pixelMap heightStride = function
    | PixelFormat.Format32bppArgb -> fun (h, w, c) -> h * heightStride + w * 4 + c
    | _                           -> fun (h, w, c) -> h * heightStride + w * 3 + c

//get data in C x H x W order from an image
let pCHW (image:Bitmap) =
    let channelStride = image.Width * image.Height
    let imageWidth = image.Width
    let imageHeight = image.Height
    let features = Array.create (imageWidth * imageHeight * 3) 0uy
    let rect = Rectangle(0, 0, imageWidth, imageHeight)
    let bitmapData = image.LockBits(rect, ImageLockMode.ReadOnly, image.PixelFormat)
    let ptr = bitmapData.Scan0
    let bytes = abs(bitmapData.Stride) * bitmapData.Height;
    let rgbValues = Array.create bytes 0uy
    let stride = bitmapData.Stride;

    // Copy the RGB values into the array.
    Marshal.Copy(ptr, rgbValues, 0, bytes)
    let mapPixel = pixelMap stride image.PixelFormat
    
    Parallel.For(0, imageHeight, fun h ->
    
        Parallel.For(0, imageWidth, fun w -> 
        
            Parallel.For(0, 3, fun c ->
                features.[channelStride * c + imageWidth * h + w] <- rgbValues.[mapPixel(h, w, c)]

            ) |> ignore
        ) |> ignore
    ) |> ignore

    image.UnlockBits(bitmapData)
    Array.map float32 features

let resize useHighQuality width height (image:Bitmap)    =
    let newImage = new Bitmap( (width:int) , (height:int) )
    newImage.SetResolution(image.HorizontalResolution, image.VerticalResolution)
    use g = Graphics.FromImage(newImage)
    g.CompositingMode <- System.Drawing.Drawing2D.CompositingMode.SourceCopy    
    if (useHighQuality) then
        g.InterpolationMode     <- InterpolationMode.HighQualityBicubic;
        g.CompositingQuality    <- CompositingQuality.HighQuality;
        g.SmoothingMode         <- SmoothingMode.HighQuality;
        g.PixelOffsetMode       <- PixelOffsetMode.HighQuality;
    else
        g.InterpolationMode     <- InterpolationMode.Default;
        g.CompositingQuality    <- CompositingQuality.Default;
        g.SmoothingMode         <- SmoothingMode.Default;
        g.PixelOffsetMode       <- PixelOffsetMode.Default;

    let attributes = new ImageAttributes();
    attributes.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
    let rect = System.Drawing.Rectangle(0, 0, width, height)
    g.DrawImage(image, rect , 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);
    newImage
