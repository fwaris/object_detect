module DetectorSettings
open OpenCvSharp
open CNTK

let searchRange = Rect(0,360,1280,670-360)

let model_path = @"..\models\detector.bin"

let srchWinHorzShift = 10
let nSrchWinSizes    = 10.//20.
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

let parallelization = 4

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


let createDetectorPool() =
    let mdl = Detector.loadFromFile model_path
    mdl::[for _ in 1 .. parallelization-1 -> mdl.Clone(ParameterCloningMethod.Clone)] 

let loadModel() = Detector.loadFromFile model_path
