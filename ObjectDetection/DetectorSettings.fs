module DetectorSettings
open OpenCvSharp

let searchRange = Rect(0,360,1280,670-360) //lower part of image for vehicle detection

let model_path = @"..\models\detector.bin"

let nSrchRows        = 20.    //number of rows of search windows
let srchWinHorzShift = 20     //horizontal shift in pixels for search win in each row
let srchRectMinHt    = 40.    //min window height
let srchRectMaxHt    = 300.   //max window height
let srchTopStart     = 65.    //top position of first row windows
let srchTopEnd       = 0.     //top position of last row windows
let srchLeftStart    = 300.   //start left postion of first row
let srchLeftEnd      = -250.  //start left position of last row

let topIncr         = (srchTopStart - srchTopEnd) / nSrchRows 
let srchRectHtIncr  = (srchRectMaxHt - srchRectMinHt) / nSrchRows 
let srchLeftIncr    = (srchLeftStart - srchLeftEnd) / nSrchRows

let srchWinTops     = [for i in srchTopStart ..  -topIncr .. srchTopEnd -> int i]
let srchWinHts      = [for i in srchRectMinHt .. srchRectHtIncr .. srchRectMaxHt -> int i]
let srchLefts       = [for i in srchLeftStart .. -srchLeftIncr .. srchLeftEnd -> int i]

//cache search windows for object detection
//the windows sizes and locations are controlled by the settings above
let srchWins = 
    Seq.zip3 srchWinTops srchLefts srchWinHts 
    |> Seq.collect (fun (t,l,h) ->
        [for l' in l .. srchWinHorzShift .. (searchRange.Width - l) ->  Rect(l',t,h,h) ]
        |> List.filter (fun r->r.Left >= 0 && r.Bottom <= searchRange.Bottom && r.Right <= searchRange.Right)
    )
    |> Seq.toArray

let loadModel() = Detector.loadFromFile model_path
