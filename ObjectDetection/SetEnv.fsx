//#I @"D:\Repos\opencvsharp\testrefs"
#I @"..\packages\OpenCvSharp3-AnyCPU.3.2.0.20170911\lib\net46"
#r @"OpenCvSharp.dll"
#r @"OpenCvSharp.Blob.dll"
#r @"OpenCvSharp.Extensions.dll" 
#r "System.Windows.Forms.DataVisualization.dll"
#r @"..\packages\FSharp.Data.2.4.2\lib\net45\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.3.20.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\FSharp.Collections.ParallelSeq.1.0.2\lib\net40\FSharp.Collections.ParallelSeq.dll"
#r @"..\packages\CNTK.GPU.2.2.0\lib\net45\x64\Cntk.Core.Managed-2.2.dll"
#r "System.Drawing"
#load "OpenCVCommon.fs"
#load "Utils.fs"
#load "Probability.fs"
#load "ObjectTracking.fs"
#load "CNTKImageUtils.fs"
#load "Detector.fs"
open System.IO
let path = Path.Combine(__SOURCE_DIRECTORY__,@"..\nativeDLLs")
System.Environment.CurrentDirectory <- path


