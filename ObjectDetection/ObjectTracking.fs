module ObjectTracking
open OpenCvSharp
open MathNet.Numerics
open System.Runtime.Remoting.Metadata.W3cXsd2001

type Track = 
    {
        Tracking        : int
        Rects           : ResizeArray<Rect>
        CtrXs           : ResizeArray<float>
        CtrYs           : ResizeArray<float>
        AvgWidth        : float
        AvgHeight       : float
        PredictedCtr    : Point2d
        XPixelsPerFrame : float
        LineIntercept   : float
        LineSlope       : float
    }

let toRect t = 
    let pX = int (t.PredictedCtr.X - (t.AvgWidth / 2.0 ))
    let pY = int (t.PredictedCtr.Y - (t.AvgHeight / 2.0))
    Rect(pX,pY,int t.AvgWidth, int t.AvgHeight)

let TRACK_MAX = 5
let FRAME_PRED_CTR_THLD = 20.

let rectCtr (r:Rect) =  Point2d(float r.X + (float r.Width) / 2.0, float r.Y + (float r.Height) / 2.0)

let append mx x (xs:ResizeArray<_>) = 
    xs.Add(x)
    if xs.Count > mx then 
        xs.RemoveAt(0) |> ignore; 
    xs

let updateAvg prevAvg prevCount newVal = (prevAvg * (float prevCount) + float newVal) / (float prevCount + 1.0)

let predictY dflt intrcpt slope x  =    
    let y = intrcpt + slope * x
    if System.Double.IsNaN y then dflt else y

//update track info with a new matching detection
let updateTrack track (detection:Rect) = 
    let ctrDtct = rectCtr detection
    let szWidth  = updateAvg track.AvgWidth track.Rects.Count detection.Width
    let szHeight = updateAvg track.AvgHeight track.Rects.Count detection.Height
    let ctrXs    = track.CtrXs |> append TRACK_MAX ctrDtct.X
    let ctrYs    = track.CtrYs |> append TRACK_MAX ctrDtct.Y
    let intrcpt,slope = Fit.Line (ctrXs.ToArray(),ctrYs.ToArray())
    let xPixelsPerFrame = ctrXs |> Seq.pairwise |> Seq.map (fun (x1,x2) -> x2-x1) |> Seq.average
    //let yPixelsPerFrame = ctrYs |> Seq.pairwise |> Seq.map( fun (y1,y2) -> y2-y1) |> Seq.average
    let predictedCntrX  = ctrDtct.X + xPixelsPerFrame
    let predictedCntrY = predictY ctrDtct.Y intrcpt slope predictedCntrX
    {track with
        Tracking        = track.Tracking + 1 |> min TRACK_MAX
        Rects           = track.Rects |> append TRACK_MAX detection
        CtrXs           = ctrXs
        CtrYs           = ctrYs
        AvgWidth        = szWidth
        AvgHeight       = szHeight
        PredictedCtr    = Point2d(predictedCntrX, predictedCntrY)
        XPixelsPerFrame = xPixelsPerFrame
        LineIntercept   = intrcpt
        LineSlope       = slope
    }

//update track info with no matching detection
let updateTrackNoDetect track =
    let predCtrX = track.PredictedCtr.X + track.XPixelsPerFrame
    let predCtrY = predictY track.PredictedCtr.Y track.LineIntercept track.LineSlope  predCtrX
    {track with
        Tracking        = track.Tracking - 1
        PredictedCtr    = Point2d(predCtrX, predCtrY)
    }

let newTrack (detection:Rect) =
    let ctr = rectCtr detection
    {
        Tracking        = 0
        Rects           = ResizeArray([detection])
        CtrXs           = ResizeArray([ctr.X])
        CtrYs           = ResizeArray([ctr.Y])
        AvgWidth        = float detection.Width
        AvgHeight       = float detection.Height
        PredictedCtr    = ctr
        XPixelsPerFrame = 0.
        LineIntercept   = 0.
        LineSlope       = 0.
    }

let private sqr x = x * x
let dist (p1:Point2d) (p2:Point2d) = sqr (p2.X - p1.X) + sqr (p2.Y - p1.Y) |> sqrt

let updateTracks tracks detections =
    if Array.isEmpty detections then
        tracks |> List.map updateTrackNoDetect
    else
        let tracks',detections' =
            (([],detections),tracks) ||> List.fold (fun (acc,detections) track ->
                if Array.isEmpty detections then
                    (acc,detections)
                else
                    let cndtDctn = detections |> Array.minBy (fun r -> dist (rectCtr r) track.PredictedCtr)
                    let cndtCtr = rectCtr cndtDctn
                    if abs(cndtCtr.X - track.PredictedCtr.X) <= FRAME_PRED_CTR_THLD &&
                        abs(cndtCtr.Y - track.PredictedCtr.Y) <= FRAME_PRED_CTR_THLD then
                        let track' = updateTrack track cndtDctn
                        let detections' =  detections |> Array.filter (fun r -> r = cndtDctn |> not)
                        track'::acc,detections'
                    else
                        let track' = updateTrackNoDetect track
                        track'::acc,detections
                )
        tracks' @ (detections' |> Seq.map newTrack |> Seq.toList)
