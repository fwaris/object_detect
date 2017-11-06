module Detector
open CNTK
open OpenCVCommon
open OpenCvSharp
open System.Runtime.InteropServices
open System.Collections.Generic

let dvc = DeviceDescriptor.GPUDevice(0)
let testFolder = @"D:\repodata\obj_detect\test_detections"
let loadFromFile (model_path:string) = Function.Load(model_path, dvc)

//score CNTK model for each search window in the given image
//Model train code in accompanying Python project
let evaluateBatch bufLen (winSize:Size) (mdl:Function) (img:Mat) (srchWins:Rect[]) =
    let inputVar = mdl.Arguments.[0]
    let outputVar = mdl.Output
    let inpMap = new Dictionary<Variable,Value>()
    let outMap = new Dictionary<Variable,Value>()
    let inputs = srchWins |> Seq.collect (fun r ->
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
    let rZip = Seq.zip srchWins rslt |> Seq.map (fun (r,ps)->r,Seq.head ps)
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

//
let detectBatch channels (winSize:Size) (mdl:Function) (img:Mat) (searchWins:Rect[]) =
    let bufLen = channels * winSize.Height * winSize.Width
    searchWins
    |> Seq.chunkBySize 2000
    |> Seq.collect (evaluateBatch bufLen winSize mdl img)

let detectBatchGreedy channels (winSize:Size) probTh (mdl:Function) (img:Mat) (searchWins:Rect[]) =
    detectBatch channels winSize mdl img searchWins
    |> Seq.filter (fun (r,p) -> p > probTh)
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
