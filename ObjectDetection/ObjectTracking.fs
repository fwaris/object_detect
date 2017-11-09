module ObjectTracking
//track vehicles across multiple frames
open OpenCvSharp
open MathNet.Numerics


let TRACK_MAX = 10 //max history for each track
let TRACK_ON  = 5  //number of trackings after which to create a track
let TRACK_OFF = 5  //tracking soft if tracking count falls to this 
let FRAME_PRED_CTR_THLD = 100.  //distance threshold to locate track in next frame
let ALPHA = 0.1 //smoothing factor

//record to hold tracking information for each vehicle
type Track = 
    {
        Tracking        : int                     //count of track rects in history
        Lost            : bool                    //false if no corresponding detection in current frame
        Rects           : ResizeArray<Rect>       //history of previous rects for this track
        CtrXs           : ResizeArray<float>      //prev center Xs
        CtrYs           : ResizeArray<float>      //prev center Ys
        AvgWidth        : float                   //avg width and height of track rects
        AvgHeight       : float                   
        PredictedCtr    : Point2d                 //predicted center of next track rect
        XPixelsPerFrame : float                   //x rate of change
        LineIntercept   : float                   // computed line intercept and slope
        LineSlope       : float
        SmoothCtr       : Point2d                 // smoothed values of center and size
        SmoothSize      : Point2d
    }

//create rect from track values
let toRect track = 
    let pX = int (track.SmoothCtr.X - (track.SmoothSize.X / 2.0 ))
    let pY = int (track.SmoothCtr.Y - (track.SmoothSize.Y / 2.0))
    Rect(pX,pY,int track.SmoothSize.X, int track.SmoothSize.Y)

//center point
let rectCtr (r:Rect) =  Point2d(float r.X + (float r.Width) / 2.0, float r.Y + (float r.Height) / 2.0)

let append mx x (xs:ResizeArray<_>) = 
    xs.Add(x)
    if xs.Count > mx then 
        xs.RemoveAt(0) |> ignore; 
    xs

//update average from
let updateAvg prevAvg prevCount newVal =
    prevAvg + (float newVal - prevAvg) / (float prevCount)

//Y value from fitted line and give X 
let predictY dflt intrcpt slope x  =    
    let y = intrcpt + (slope * x)
    if System.Double.IsNaN y then dflt else y //used default value if unable to predict

//update track info with a new matching detection
let updateTrack track (detection:Rect) = 
    let ctrDtct = rectCtr detection
    let szWidth  = updateAvg track.AvgWidth track.Rects.Count detection.Width
    let szHeight = updateAvg track.AvgHeight track.Rects.Count detection.Height
    let ctrXs    = track.CtrXs |> append TRACK_MAX ctrDtct.X
    let ctrYs    = track.CtrYs |> append TRACK_MAX ctrDtct.Y
    let intrcpt,slope = Fit.Line (ctrXs.ToArray(),ctrYs.ToArray()) //fit line across history centers
    //let intrcpt' = track.LineIntercept * 0.85 + intrcpt * 0.15
    //let slope' = track.LineSlope * 0.85 + slope * 0.15
    let xPixelsPerFrame = ctrXs |> Seq.pairwise |> Seq.map (fun (x1,x2) -> x2-x1) |> Seq.average
    //let yPixelsPerFrame = ctrYs |> Seq.pairwise |> Seq.map( fun (y1,y2) -> y2-y1) |> Seq.average
    let predCtrX  = ctrDtct.X + xPixelsPerFrame
    let predCtrY = predictY ctrDtct.Y intrcpt slope predCtrX
    let smoothX = predCtrX * ALPHA + (track.SmoothCtr.X * (1. - ALPHA))
    let smoothY = predCtrY * ALPHA + (track.SmoothCtr.Y * (1. - ALPHA))
    let smoothSzW = szWidth * ALPHA + (track.SmoothSize.X * (1. - ALPHA))
    let smoothSzH = szHeight * ALPHA + (track.SmoothSize.Y * (1. - ALPHA))
    {track with
        Tracking        = track.Tracking + 1 |> min TRACK_MAX
        Lost            = false
        Rects           = track.Rects |> append TRACK_MAX detection
        CtrXs           = ctrXs
        CtrYs           = ctrYs
        AvgWidth        = szWidth
        AvgHeight       = szHeight
        PredictedCtr    = Point2d(predCtrX, track.PredictedCtr.Y)
        XPixelsPerFrame = xPixelsPerFrame
        LineIntercept   = intrcpt
        LineSlope       = slope
        SmoothCtr       = Point2d(smoothX, smoothY)
        SmoothSize      = Point2d(smoothSzW,smoothSzH)
    }

//update track info with no matching detection (use regression line)
let updateTrackNoDetect track =
    let predCtrX = track.PredictedCtr.X + track.XPixelsPerFrame 
    let predCtrY = predictY track.PredictedCtr.Y track.LineIntercept track.LineSlope  predCtrX 
    let smoothX = predCtrX * ALPHA + (track.SmoothCtr.X * (1. - ALPHA))
    let smoothY = predCtrY * ALPHA + (track.SmoothCtr.Y * (1. - ALPHA))
    {track with
        Tracking        = track.Tracking - 1
        Lost            = true
        PredictedCtr    = Point2d(predCtrX, track.PredictedCtr.Y)
        SmoothCtr       = Point2d(smoothX, smoothY)     
    }

let newTrack (detection:Rect) =
    let ctr = rectCtr detection
    let w = float detection.Width
    let h = float detection.Height
    {
        Tracking        = 0
        Lost            = false
        Rects           = ResizeArray([detection])
        CtrXs           = ResizeArray([ctr.X])
        CtrYs           = ResizeArray([ctr.Y])
        AvgWidth        = w
        AvgHeight       = h
        PredictedCtr    = ctr
        XPixelsPerFrame = 0.
        LineIntercept   = 0.
        LineSlope       = 0.
        SmoothCtr       = ctr
        SmoothSize      = Point2d(w,h)
    }

let private sqr x = x * x

let dist (p1:Point2d) (p2:Point2d) = sqr (p2.X - p1.X) + sqr (p2.Y - p1.Y) |> sqrt

let cdist (r1:Rect) (r2:Rect)  = 
    dist (Point2d(float r1.X, float r1.Y)) (Point2d(float r2.X, float r2.Y)) +
    dist (Point2d(r1.X + r1.Width |> float, r1.Y + r1.Height |> float)) (Point2d(r2.X + r2.Width |> float, r2.Y + r2.Height |> float)) 

let tDist (pCtr:Point2d) (r1:Rect) (r2:Rect) =
    let d1 = dist pCtr (rectCtr r2)
    let d2 = cdist r1 r2
    d1 + 0.25 * d2

//update all tracks from the detections 
//in the current frame
let updateTracks tracks detections =
    let tracks',detections' =
        (([],detections),tracks) ||> List.fold (fun (acc,detections) track ->
            if Array.isEmpty detections then
                ((updateTrackNoDetect track) :: acc,detections)
            else
                //let cndtDctn = detections |> Array.minBy (fun r -> dist (rectCtr r) track.PredictedCtr)
                //let cndtDctn = detections |> Array.minBy (fun r -> cdist r (toRect track)) 
                let cndtDctn = detections |> Array.minBy (fun r -> tDist track.PredictedCtr (toRect track) r)
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
