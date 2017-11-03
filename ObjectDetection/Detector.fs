module Detector
open CNTK
open OpenCVCommon
open OpenCvSharp
open System.Runtime.InteropServices
open System.Collections.Generic
open FSharp.Collections.ParallelSeq
open System.IO

let dvc = DeviceDescriptor.GPUDevice(0)
let testFolder = @"D:\repodata\obj_detect\test_detections"
let loadModel (model_path:string) = Function.Load(model_path, dvc)

let evaluate bufLen (winSize:Size) (img:Mat) (mdl:Function,r:Rect) =
    use m = img.SubMat(r)
    use m' = new Mat()
    Cv2.Resize(!> m, !> m', winSize)
    let buff = Array.create bufLen 0uy
    Marshal.Copy(m'.Data,buff,0,buff.Length)
    let inputVar = mdl.Arguments.[0]
    let outputVar = mdl.Output
    let inpMap = new Dictionary<Variable,Value>()
    let outMap = new Dictionary<Variable,Value>()
    inpMap.Add(inputVar,Value.CreateBatch(inputVar.Shape,buff |> Seq.map float32, dvc))
    outMap.Add(outputVar,null)
    mdl.Evaluate(inpMap,outMap,dvc)
    let rslt = outMap.[outputVar].GetDenseData<float32>(outputVar)
    let prob = rslt |> Seq.head |> Seq.head
    let rs = if  prob > 0.7f then Some r else None
    if rs.IsSome then
        let fn = Path.Combine(testFolder,sprintf "r%d%d%d%d_%s.png" r.Left r.Top r.Width r.Height ((string prob).Replace(".","_")))
        m'.SaveImage(fn) |> ignore
    rs

let detect channels (winSize:Size) (mdlPool:Function list) (img:Mat) (searchWins:Rect[]) =
    if searchWins.Length % mdlPool.Length <> 0 then failwith "no. of srch wins must be evenly divisible by mdl pool size"
    let bufLen = channels * winSize.Height * winSize.Width
    searchWins 
    |> Seq.chunkBySize mdlPool.Length
    |> Seq.collect (fun chnk -> 
        (mdlPool,chnk) 
        ||> Seq.zip
        |> PSeq.map (evaluate bufLen winSize img))
    |> Seq.choose (fun r->r)

let testDetect (img:Mat) (mdl:Function) = 
    let bufLen = img.Size(0) * img.Size(1) * img.Channels()
    let buff = Array.create bufLen 0uy
    Marshal.Copy(img.Data,buff,0,buff.Length)
    let inputVar = mdl.Arguments.[0]
    let outputVar = mdl.Output
    let inpMap = new Dictionary<Variable,Value>()
    let outMap = new Dictionary<Variable,Value>()
    inpMap.Add(inputVar,Value.CreateBatch(inputVar.Shape,buff |> Seq.map float32, dvc))
    outMap.Add(outputVar,null)
    mdl.Evaluate(inpMap,outMap,dvc)
    let rslt = outMap.[outputVar].GetDenseData<float32>(outputVar)
    let prob = rslt |> Seq.head |> Seq.head
    prob
