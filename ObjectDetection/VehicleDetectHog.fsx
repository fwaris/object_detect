#load "SetEnv.fsx"
open OpenCvSharp
open OpenCVCommon
open System
open System.IO
open Utils
open ObjectTracking

let trainWinSize = Size( 64, 64 )  

//normalize frame or image before feature extraction
let normalizeFrame (img:Mat) =
    let normd = new Mat()
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2YUV)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2HLS_FULL)
    // ** alternate normalization methods tried
    Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2YCrCb)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2HSV)
    //Cv2.AddWeighted(!> img, 0.5, !> hsv, 0.5, 0.30, !> normd)
    //img.CopyTo(normd)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2GRAY)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2HLS)
    normd
    //img

//draw rectangles on an image for the detected objects
//strength of detection is indicated by shade of green
//of the rectangle outline
let drawRectsWts (detections,weights:float[]) (img:Mat) =
    detections |> Array.iteri (fun i boundingBox ->
        let clr = Scalar(0.,weights.[i] * weights.[i] * 200., 0.)
        Cv2.Rectangle(img,boundingBox,clr,img.Cols/400 + 1)
        printfn "weight %d %f" i weights.[i])

//alternate draw rectangle method  
let drawRects detections (img:Mat) =
    detections |> Seq.iteri (fun i d ->
        let clr = Scalar(125.,200., 0.)
        Cv2.Rectangle(img,d,clr,1))


//Create a heat map from the given detections.
//It increments a counter for each pixel covered by a rectangle
let heatmapRects (sz:Size) (detections:Rect[]) =
    let plane = Array2D.create sz.Height sz.Width   0uy
    let ds = detections |> Array.filter (fun r->r.Width > 5 && r.Height > 5)
    for r in ds do
        for i in r.Y..r.Y+r.Height-1 do
            for j in r.X..r.X+r.Width-1 do
                let v = plane.[i,j]
                //let w = weightScale r
                plane.[i,j] <- v + 2uy|> min 255uy
    plane

//Converge multiple overlapping detections into a single bounding box
//One actual object may be represented by many detections (rectangles).
//This method merges mutliple overlapping rectangles into a single one.
//The steps are as follows:
// a) create a 'heat map' from the detections
// b) find contours around 'hot' regions using OpenCV FindContours method
// c) find a bounding box from each contour using OpenCV BoundingRec method
//The threshold value is used to sharpen the contour boundaries
let findBoundingBoxes (threshold:float) (img:Mat) (rects:Rect[]) =
    let m2a = heatmapRects (img.Size()) rects                           //heat map
    use m2m = new Mat([img.Height;img.Width],MatType.CV_8UC1,m2a)       //convert heat map to gray scale image
    Cv2.InRange(!>m2m,Scalar(threshold),Scalar(255.),!>m2m)             //threshold image to sharpen contours
    let mutable pts : Point[][] = null
    let mutable hi : HierarchyIndex[] = null
    Cv2.FindContours(!>m2m,&pts,&hi,RetrievalModes.External,ContourApproximationModes.ApproxSimple)  //find contours
    let boundings = pts |> Array.map(fun pts->Cv2.BoundingRect(pts))                                 //find bounding boxes from contours
    boundings

let aspectRatio (r:Rect) = min (float r.Width / float r.Height) (float r.Height / float r.Width)

let drawTrack track (img:Mat) =
    let rect = toRect track
    //printfn "%A" rect
    Cv2.Rectangle(img,rect,Scalar(255.,150.,255.),5)
    Cv2.PutText(!>img,sprintf "%d" track.Tracking, Point(rect.X + 2, rect.Y + 2), HersheyFonts.HersheyPlain, 8., Scalar(255.,0.,0.))

let repositionRect (r:Rect) =  Rect(r.X, DetectorSettings.searchRange.Y + r.Y, r.Width, r.Height )
   
//input and output video files paths
let v_prjctP = @"D:\repodata\obj_detect\project_video.mp4"
let v_prjctT = @"D:\repodata\obj_detect\test_video.mp4"
let v_prjctPOut = @"D:\repodata\obj_detect\project_video_out.mp4"
let v_prjctTOut = @"D:\repodata\obj_detect\test_video_out.mp4"

//process the input video file to detect objects
//and write an annotated video to the output
//(the annotations are bounding boxes over the detected objects)
let testVideoDetect (v_prjct:string) (v_out:string) =
    let mdl = DetectorSettings.loadModel() 
    let probTh = 0.70f
    let cntrTh = 1.
    let overlapTh = 0.50
    let mutable tracks = []
    //video processing
    use clipIn = new VideoCapture(v_prjct)
    let imgSz = new Size(clipIn.FrameWidth,clipIn.FrameHeight)
    use clipOut = new VideoWriter()
    clipOut.Open(v_out,FourCC.DIVX,clipIn.Fps,imgSz)
    if not(clipOut.IsOpened()) then failwith "file not opened"
    let r = ref 0
    let folder = @"D:\repodata\obj_detect\test2"        //folder to output individual annotated images - in addition to creating a video
    if Directory.Exists folder |> not then Directory.CreateDirectory folder |> ignore
    while clipIn.Grab() do
        try
            use m = clipIn.RetrieveMat()
            use n = normalizeFrame m                           //normalize input frame
            use roi = n.SubMat(DetectorSettings.searchRange)
            let hits = Detector.detectBatchGreedy 
                            3 
                            trainWinSize 
                            probTh 
                            mdl 
                            roi 
                            DetectorSettings.srchWins
            let hitNms = NMS.nms overlapTh hits |> List.map repositionRect
            let rects = hitNms |> Seq.toArray
            let detections = findBoundingBoxes cntrTh m rects  |> Array.filter (fun r->aspectRatio r > 0.30) //merge overlapping detections
            tracks <- updateTracks tracks detections |> List.filter (fun t->t.Tracking >= 0)
            tracks |> List.iter (fun t -> if t.Tracking > 3 then drawTrack t m)
            detections |> Array.iter (fun b -> Cv2.Rectangle(m,b,Scalar(0.,255.,255.),2))  //draw bounding boxes over detected objects
            clipOut.Write(m)
            let fn = Path.Combine(folder,sprintf "th%d.jpg" !r)
            m.SaveImage(fn) |> ignore
            r := !r + 1
            printfn "th %d" !r
            m.Release()
            n.Release()
            roi.Release()
        with ex ->
            printfn "frame miss %d %s" !r ex.Message
    clipIn.Release()
    clipOut.Release()
;;

//Utility method to test the detector on a single image
let testDetector() =
    let mdl = DetectorSettings.loadModel() 
    let probTh = 0.70f
    let cntrTh = 1.
    let overlapTh = 0.50
    //let file = @"D:\repodata\adv_lane_find\imgs\img304.jpg"
    //let file = @"D:\repodata\adv_lane_find\imgs\img495.jpg"
    let file = @"D:\repodata\adv_lane_find\imgs\img588.jpg"
    //let file = @"D:\repodata\adv_lane_find\imgs\img209.jpg"
    //let file = @"D:\repodata\adv_lane_find\imgs\img280.jpg"
    let img = Cv2.ImRead(file)
    let normd = normalizeFrame img
    //let wm = detector hd normd
    //let rects = fst wm
    //let heatMap = heatmapRects (img.Size()) (fst wm)
    let roi = normd.SubMat(DetectorSettings.searchRange)
    Directory.GetFiles(Detector.testFolder) |> Array.iter File.Delete
    let hits = Detector.detectBatchGreedy 
                    3 
                    trainWinSize 
                    probTh 
                    mdl
                    roi 
                    DetectorSettings.srchWins

    hits |> Seq.sortByDescending snd |> Seq.iter (fun (r,f) -> printfn "(Rect(%d,%d,%d,%d),%ff)" r.X r.Y r.Width r.Height f)
    let hitNms = NMS.nms overlapTh hits |> List.map repositionRect
    NMS.overlap hitNms.[2] hitNms.[0]
    let rects = hitNms |> Seq.toArray
    let heatMap = heatmapRects (img.Size()) rects
    use heatMapGray = new Mat([img.Height;img.Width],MatType.CV_8UC1,heatMap)
    use heatMapTh = heatMapGray.EmptyClone()
    Cv2.InRange(!>heatMapGray,Scalar(cntrTh),Scalar(255.),!>heatMapTh)
    let mutable pts : Point[][] = null
    let mutable hi : HierarchyIndex[] = null
    Cv2.FindContours(!>heatMapTh,&pts,&hi,RetrievalModes.External,ContourApproximationModes.ApproxSimple)
    let pts2 : Collections.Generic.IEnumerable<Collections.Generic.IEnumerable<Point>> = pts|>Seq.map(fun p-> p |> Array.toSeq)
    let ctrs = img.EmptyClone()
    pts2 |> Seq.iteri (fun i _ -> Cv2.DrawContours(!>ctrs, pts2,i,Scalar(255.,255.,255.), hierarchy=hi))
    let boundings = pts |> Array.map(fun pts->Cv2.BoundingRect(pts))
    let b1 = img.Clone()
    boundings |> Array.iter(fun b -> Cv2.Rectangle(b1,b,Scalar(0.,255.,0.)))
    //boundings |> Array.iter(fun b -> checkBdetect img b)
    let rawRects = img.Clone()
    drawRects rects rawRects

    win "rawRects" rawRects
    win "heatMapGray" heatMapGray
    win "heatMapTh" heatMapTh
    win "ctrs" ctrs
    win "i" img
    win "b1" b1

    let folder = @"D:\repodata\obj_detect\detect_steps"
    if Directory.Exists folder |> not then Directory.CreateDirectory folder |> ignore
    img.SaveImage(folder + @"\input.png")
    rawRects.SaveImage(folder + @"\rawRects.png")
    heatMapGray.SaveImage(folder + @"\heatMapGray.png")
    heatMapTh.SaveImage(folder + @"\heatMapTh.png")
    ctrs.SaveImage(folder + @"\ctrs.png")
    b1.SaveImage(folder + @"\output.png")

(*
testVideoDetect v_prjctP v_prjctPOut;;
testVideoDetect v_prjctT v_prjctTOut
*)
