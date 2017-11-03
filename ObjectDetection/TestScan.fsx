#load "SetEnv.fsx"
#r @"..\packages\CNTK.GPU.2.2.0\lib\net45\x64\Cntk.Core.Managed-2.2.dll"
open CNTK
open OpenCVCommon
open OpenCvSharp
open System.Runtime.InteropServices
open System.Collections.Generic
open Utils
open System.IO

let imgFile = @"D:\repodata\adv_lane_find\imgs\img126.jpg"

let img = Cv2.ImRead(imgFile)

let subMatScale (r:Rect) (sz:Size) (buff:byte[]) (img:Mat) =
    use subMat = img.SubMat(r)
    use m2 = new Mat()
    Cv2.Resize(!>subMat, !> m2, sz)
    Marshal.Copy(m2.Data,buff,0,buff.Length)

let sz = Size(64,64)

let searchRange = Rect(0,360,1280,670-360)

//let sr = img.SubMat(searchRange)
//sr.SaveImage( @"D:\repodata\obj_detect\scanning\searchRange.png")
//win "sr" sr

let srchWinHorzShift = 10
let nSrchWinSizes    = 30.
let srchRectMinHt    = 20.
let srchRectMaxHt    = 310.
let srchTopStart     = 65.
let srchTopEnd       = 0.
let srchLeftStart    = 300.
let srchLeftEnd      = -250.

let topIncr         = (srchTopStart - srchTopEnd) / nSrchWinSizes 
let srchRectHtIncr  = (srchRectMaxHt - srchRectMinHt) / nSrchWinSizes 
let srchLeftIncr    = (srchLeftStart - srchLeftEnd) / nSrchWinSizes

let srchWinTops     = [for i in srchTopStart ..  -topIncr .. srchTopEnd -> int i]
let srchWinHts      = [for i in srchRectMinHt .. srchRectHtIncr .. srchRectMaxHt -> int i]
let srchLefts       = [for i in srchLeftStart .. -srchLeftIncr .. srchLeftEnd -> int i]

let parallelization = 8
let srchWins = 
    let winList = 
        Seq.zip3 srchWinTops srchLefts srchWinHts 
        |> Seq.collect (fun (t,l,h) ->
            [for l' in l .. srchWinHorzShift .. (searchRange.Width - l) ->  Rect(l',t,h,h) ]
            |> List.filter (fun r->r.Left >= 0 && r.Bottom <= searchRange.Bottom && r.Right <= searchRange.Right)
        )
        |> Seq.toArray
    let m = [for i in winList.Length .. -1 .. 0 -> i] |> List.find(fun l -> l % parallelization = 0) 
    winList |> Array.skip (winList.Length - m) //evenly divisible by parallelization factor

let ts = srchWins |> Array.filter (fun r-> r.Left < 10)
    
Directory.GetFiles(Detector.testFolder) |> Array.iter File.Delete
let model_path = @"..\models\detector.bin"
if srchWins.Length % parallelization <> 0 then failwith "parallelization error"
let mdl = Detector.loadModel model_path
let pool  = mdl::[for _ in 1 .. parallelization-1 -> mdl.Clone(ParameterCloningMethod.Clone)] 
let sr = img.SubMat(searchRange)
let detections = Detector.detect 3 sz pool sr srchWins |> Seq.toArray

let testSr = img.Clone().SubMat(searchRange)
detections |> Seq.iter (fun r-> Cv2.Rectangle(testSr, r, Scalar(123.,5.,25.),1))
ts |> Seq.rev |> Seq.filter (fun r-> r.Height > 200) |> Seq.iter (fun r ->  Cv2.Rectangle(testSr,r,Scalar(255. - float r.Height * 1.7 |> max 0.,  215.,  float r.Height * 0.7 |> min 255.),1))
win "testSr" testSr

let t1 = @"D:\repodata\obj_detect\vehicles\wc\i67.png"
let t1M = Cv2.ImRead(t1)
t1M.Size(0),t1M.Size(1),t1M.Channels()
let p = Detector.testDetect t1M mdl


//testSr.SaveImage( @"D:\repodata\obj_detect\scanning\searchRangeRectsAll.png")
//win "sr" sr
//win "img" img