//#I @"D:\Repos\opencvsharp\testrefs"
#I @"..\packages\OpenCvSharp3-AnyCPU.3.2.0.20170911\lib\net46"
#r @"OpenCvSharp.dll"
#r @"OpenCvSharp.Blob.dll"
#r @"OpenCvSharp.Extensions.dll" 
#r "System.Windows.Forms.DataVisualization.dll"
#r @"..\packages\FSharp.Data.2.4.2\lib\net45\FSharp.Data.dll"
#load "OpenCVCommon.fs"
#load "Utils.fs"
#load "Probability.fs"
open System.Runtime.InteropServices
open System.IO
let path = Path.Combine(__SOURCE_DIRECTORY__,@"..\packages\OpenCvSharp3-AnyCPU.3.2.0.20170911\NativeDlls\x64")
System.Environment.CurrentDirectory <- path


