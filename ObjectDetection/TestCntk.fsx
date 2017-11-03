#load "SetEnv.fsx"
#r @"..\packages\CNTK.GPU.2.2.0\lib\net45\x64\Cntk.Core.Managed-2.2.dll"
open CNTK
open OpenCVCommon
open OpenCvSharp
open System.Runtime.InteropServices
open System.Collections.Generic
open System.Drawing

let dvc = DeviceDescriptor.GPUDevice(0)
let model_path = @"C:\Users\fwaris\Documents\Visual Studio 2017\Projects\PythonClassifierApplication2\my_cntk_model"
let mdl = Function.Load(model_path, dvc)
let inputVar = mdl.Arguments.[0]
//let img = Cv2.ImRead(@"D:\repodata\obj_detect\non-vehicles\nonudcars\i41.png")
let test1 = "D:/repodata/obj_detect/nonudcars/i8.png"
let test2 = "D:/repodata/obj_detect/vehicles/udcars/i7.png"

let img = Cv2.ImRead(test2)
let ms = img.Split()
let b = Array.create (64*64) 0uy
let g = Array.create b.Length 0uy
let r = Array.create b.Length 0uy
Marshal.Copy(ms.[0].Data,b,0,b.Length)
Marshal.Copy(ms.[1].Data,g,0,b.Length)
Marshal.Copy(ms.[2].Data,r,0,b.Length)
let imgData = Seq.collect (fun x->x) [b; g; r] |> Seq.map float32
let m1 = Cv2.ImRead(test1)
let m2 = Cv2.ImRead(test2)
let t1 = Array.create (64*64*3) 0uy
let t2 = Array.create (64*64*3) 0uy
Marshal.Copy(m1.Data,t1,0,t1.Length)
Marshal.Copy(m2.Data,t2,0,t2.Length)
let tf1 = t1 |> Seq.map float32
let tf2 = t2 |> Seq.map float32

//let im2 = new Bitmap(Bitmap.FromFile(test1))
//let imgData2 = CNTKImageUtils.pCHW im2
//Seq.zip imgData imgData2 |> Seq.iteri (fun i (a,b) -> if a<>b then printfn "%d" i)

let inputVal = Value.CreateBatch(inputVar.Shape,tf1,dvc)
let outputVar = mdl.Output
let inpMap = new Dictionary<Variable,Value>()
let outMap = new Dictionary<Variable,Value>()
inpMap.Add(inputVar,inputVal)
outMap.Add(outputVar,null)
mdl.Evaluate(inpMap,outMap,dvc)
let rslt = outMap.[outputVar].GetDenseData<float32>(outputVar)
rslt |> Seq.head |> Seq.head
