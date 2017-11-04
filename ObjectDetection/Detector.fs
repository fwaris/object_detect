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
let loadFromFile (model_path:string) = Function.Load(model_path, dvc)

let evaluate bufLen (winSize:Size) probTrhld (img:Mat) (mdl:Function,r:Rect) =
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
    let rs = if  prob > probTrhld then Some r else None
    //if rs.IsSome then
    //    let fn = Path.Combine(testFolder,sprintf "r%d%d%d%d_%s.png" r.Left r.Top r.Width r.Height ((string prob).Replace(".","_")))
    //    m'.SaveImage(fn) |> ignore
    rs

let detect channels (winSize:Size) probTh (mdlPool:Function list) (img:Mat) (searchWins:Rect[]) =
    if searchWins.Length % mdlPool.Length <> 0 then failwith "no. of srch wins must be evenly divisible by mdl pool size"
    let bufLen = channels * winSize.Height * winSize.Width
    searchWins 
    |> Seq.chunkBySize mdlPool.Length
    |> Seq.collect (fun chnk -> 
        (mdlPool,chnk) 
        ||> Seq.zip
        |> PSeq.map (evaluate bufLen winSize probTh img))
    |> Seq.choose (fun r->r)

let evaluateBatch bufLen (winSize:Size) (mdl:Function) (img:Mat) (chnk:Rect[]) =
    let inputVar = mdl.Arguments.[0]
    let outputVar = mdl.Output
    let inpMap = new Dictionary<Variable,Value>()
    let outMap = new Dictionary<Variable,Value>()
    let inputs = chnk |> Seq.collect (fun r ->
        use m = img.SubMat(r)
        use m' = new Mat()
        Cv2.Resize(!> m, !> m', winSize)
        let buff = Array.create bufLen 0uy
        Marshal.Copy(m'.Data,buff,0,buff.Length)
        buff |> Seq.map float32)
    let batch = Value.CreateBatch(inputVar.Shape, inputs, dvc)
    inpMap.Add(inputVar,batch)
    outMap.Add(outputVar,null)
    mdl.Evaluate(inpMap,outMap,dvc)
    let rslt = outMap.[outputVar].GetDenseData<float32>(outputVar)
    let rZip = Seq.zip chnk rslt |> Seq.map (fun (r,ps)->r,Seq.head ps)
    //debugging code - saves each image patch with prob score
    //rZip |> Seq.iter (fun (r,prob) ->
    //    if prob > 0.75f then
    //        let fn = Path.Combine(testFolder,sprintf "r%d%d%d%d_%s.png" r.Left r.Top r.Width r.Height ((string prob).Replace(".","_")))
    //        use m = img.SubMat(r)
    //        use m' = new Mat()
    //        Cv2.Resize(!> m, !> m', winSize)
    //        m'.SaveImage(fn) |> ignore
    //)
    rZip

let detectBatch channels (winSize:Size) probTh (mdl:Function) (img:Mat) (searchWins:Rect[]) =
    let bufLen = channels * winSize.Height * winSize.Width
    searchWins
    |> Seq.chunkBySize 1000
    |> Seq.collect (evaluateBatch bufLen winSize mdl img)
    |> Seq.filter (fun (r,p) -> p > probTh)
    |> Seq.map fst
    |> Seq.toArray

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
