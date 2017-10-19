#load "SetEnv.fsx"
open OpenCvSharp
open OpenCVCommon
open System
open System.IO
open SetEnv
open Utils
open OpenCvSharp.ML
open System.Runtime.InteropServices
open Probability

//windows size over which the Histogram of Gradient (HOG) featuers are extracted
let trainWinSize = Size( 64, 64 )  

//SVM file - trained SVM is stored here and used later in prediction
let svmFile = @"D:\repodata\obj_detect\veh_detect.yml" 

//create OpenCV HOG Descriptor 
//see OpenCV documentation for details
let hog (sz:Size) =
    let hd = new HOGDescriptor()
    hd.WinSize <- sz
    hd.BlockSize <- Size(16.,16.)
    hd.BlockStride <- Size(8.,8.)
    hd.CellSize <- Size(8.,8.)
    hd.Nbins <- 9
    hd.DerivAperture <- 2
    hd.WinSigma <- 2.0 //1->0.9;1.5->0.92;2->0.94;3.5->0.94
    hd.HistogramNormType <- HistogramNormType.L2Hys
    hd.L2HysThreshold <- 0.2  //0.15 -> .923; 0.10 -> .912; 0.2->.923; 0.25->.916
    hd.GammaCorrection <- false
    //hd.GetDescriptorSize()
    hd

//normalize frame or image before feature extraction
let normalizeFrame (img:Mat) =
    let normd = new Mat()
    img.CopyTo(normd)
    // ** alternate normalization methods tried
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2GRAY)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2HLS)
    //Cv2.CvtColor(!>img,!>normd,ColorConversionCodes.BGR2HSV)
    normd

//normalize and compute HOG features for the input image
let compDesc (hd:HOGDescriptor) (img:Mat) =
    use normd = normalizeFrame img
    let pts = hd.Compute(normd)
    normd.Release()
    pts

//compute HOG features for an image file
let getFeatures (hd:HOGDescriptor) imagePath =
    //printfn "%s" imagePath
    use img = Cv2.ImRead imagePath
    let pts = compDesc hd img
    img.Release()
    pts

//positive and negative images for training (all sized to 64x64)
let postiveImagesFolder = @"D:\repodata\obj_detect\vehicles"
let negativeImagesFolder = @"D:\repodata\obj_detect\non-vehicles"
let posTrain = Directory.GetFiles(postiveImagesFolder,"*.png",SearchOption.AllDirectories)
let negTrain = Directory.GetFiles(negativeImagesFolder,"*.png",SearchOption.AllDirectories)

Array.shuffle posTrain //shuffle loaded image paths to randomize input
Array.shuffle negTrain

//the percent of data to use for training
//(over 40K images in input folders so don't need to use all of them, to save time)
let testPct = 0.3    

let NTrainPos = (float posTrain.Length * testPct) |> int //number of pos and neg images
let NTrainNeg = (float negTrain.Length *testPct) |> int

//Train an SVM (uses the OpenCV SVM class)
//SVM is trained on HOG features extracted using the provided HOGDescriptor instance
//from the supplied positive and negative images (given as arrays of paths to images)
//Saves trained SVM to the given output file
let trainSvm (hd:HOGDescriptor) posTrain negTrain svmFile = 
    
    //extract HOG features from images
    let ftrsPos = posTrain |> Seq.map (getFeatures hd) |> Seq.take NTrainPos
    let ftrsNeg = negTrain |> Seq.map (getFeatures hd) |> Seq.take NTrainNeg
    let head  = ftrsPos |> Seq.head //keep around first feature array for later
    let features = Seq.append ftrsPos ftrsNeg |> Seq.toArray

    //construct labels [-1,+1]
    let labels = 
        Seq.append 
            [for _ in 1..Seq.length ftrsPos -> 1]  
            [for _ in 1..Seq.length ftrsNeg -> -1] |> Seq.toArray
    
    //shuffle train data features and labels
    Array.shuffle2 features labels

    //convert the feature set (an array of arrays) into a flat array
    let flatFeatures = features |> Array.collect (fun x -> x)

    //construct OpenCV Mat structures for features and labels
    let ftrMat = new Mat([flatFeatures.Length / head.Length; head.Length],MatType.CV_32FC1,flatFeatures)
    let lblMat = new Mat([labels.Length],MatType.CV_32SC1,labels)
 
    //create and configure OpenCV SVM class
    //see open CV documentation for details 
    //settings are specific to the SVM.Type (many SVM types were tried)
    let svm = SVM.Create()
    svm.Coef0 <- 0.0
    svm.Degree <- 2.
    svm.TermCriteria <- (TermCriteria(CriteriaType.Eps + CriteriaType.MaxIter,2000,0.001))
    svm.Gamma <- 0.
    svm.KernelType <- SVM.KernelTypes.Linear
    svm.Nu <- 0.5
    svm.P <- 0.01
    svm.C <- 0.01

    //use epsilon 'regression' SVM type for object detection, 
    //following OpenCV object detection sample
    svm.Type <- SVM.Types.EpsSvr 
    let r = svm.Train(!> ftrMat, SampleTypes.RowSample, !> lblMat)
    svm.Save(svmFile)


//test trained SVM on out-of-sample data
let testPrediction() =
    let hd = hog trainWinSize
    let svmT = SVM.Load(svmFile)
    let ftrsPos = posTrain |> Seq.map (getFeatures hd) |> Seq.skip NTrainPos |> Seq.take 300
    let ftrsNeg = negTrain |> Seq.map (getFeatures hd) |> Seq.skip NTrainNeg |> Seq.take 300
    let ftrsTest = Seq.append ftrsPos ftrsNeg |> Seq.collect (fun x -> x) |> Seq.toArray
    let h  = ftrsPos |> Seq.head
    let ftrSz = [ftrsTest.Length / h.Length; h.Length]
    let test = new Mat(ftrSz,MatType.CV_32FC1,ftrsTest)
    let y_act = 
        Seq.append 
            [for _ in 1..Seq.length ftrsPos -> +1]  
            [for _ in 1..Seq.length ftrsNeg -> -1] |> Seq.toArray
    let out = new Mat()
    svmT.Predict(!>test,!> out)
   
    //let y_pred = [for i in 0..out.Rows-1 ->  if out.Get<int>(i,0) > 0 then 1 else -1]
    let y_pred = [for i in 0..out.Rows-1 ->  if out.Get<float>(i,0) > 0. then 1 else -1]
    //let y_predProb = [for i in 0..out.Rows-1 ->  out.Get<float>(i,0) ]
    //y_predProb |> Seq.min, y_predProb |> Seq.max
    let acc = Seq.zip y_act y_pred |> Seq.map (fun (a,p) -> if a=p then 1. else 0.) |> Seq.sum |> fun s-> s/ float y_pred.Length
    acc

//An OpenCV HOGDescriptor instance can be configured with a 
//trained OpenCV SVM for object detection 
//Thus equipped, the HOGDescriptor can use an efficient
//multi-scale, sliding-window search to detect objects in a given image
//this method configures a HOGDescriptor with the given trained SVM
//It follows an OpenCV sample for object detection
let setDetector (hd:HOGDescriptor) (trainedSvm:SVM) =
    use sv = trainedSvm.GetSupportVectors()
    use alpha = new Mat()
    use svidx = new Mat()
    let rho = float32 <| trainedSvm.GetDecisionFunction(0, !> alpha, !> svidx)
    assert(alpha.Total() = 1L && svidx.Total() = 1L && sv.Rows = 1)
    assert(alpha.Type() = MatType.CV_64FC1 && alpha.At<float>(0) = 1.0
    || alpha.Type() = MatType.CV_32FC1 && alpha.At<float32>(0) =  1.0f)
    assert(sv.Type() = MatType.CV_32FC1)
    let vs = Array.create (sv.Cols+1) 0.0f
    Marshal.Copy(sv.Data,vs,0,sv.Cols)
    //vs.[sv.Cols] <- rho
    vs.[sv.Cols] <- -rho
    hd.SetSVMDetector vs
    ()

//Use the given HOGDescriptor (configured with an SVM detector)
//to perform a mult-scale, sliding-window search
//to detect objects in the given image
//This method returns an array of bounding boxes for the detected
//objects and the associated weights (which signify the strengths of detections)
// - padding is used to pad the search windows 
// - threshold is used to filter out detections whose weight is below the threshold
//   (threshold can be used to reduce false positives - a good value is found by trial)
// - group is a factor for nested bounding boxes but is not really used for this application
let detect (padding:Size) threshold group (hd:HOGDescriptor)  (img:Mat) =
    let mutable weights : float[] = null
    let ws  : Nullable<Size> = !> Size(8.,8.)
    let padding : Nullable<Size> = !> padding
    //hog.detectMultiScale(segment, found, 0.0, winStride, padding, 1.01, 0.1);
    let detections = hd.DetectMultiScale(img,
                            &weights, 
                            hitThreshold=threshold,
                            winStride=ws,
                            padding=padding,
                            scale=1.1,
                            groupThreshold = group)
    detections,weights 

//draw rectangles on an image for the detected objects
//strength of detection is indicated by shade of green
//of the rectangle outline
let drawRects (detections,weights:float[]) (img:Mat) =
    detections |> Array.iteri (fun i boundingBox ->
        let clr = Scalar(0.,weights.[i] * weights.[i] * 200., 0.)
        Cv2.Rectangle(img,boundingBox,clr,img.Cols/400 + 1)
        printfn "weight %d %f" i weights.[i])

//alternate draw rectangle method  
let drawRectsSmpl (detections,weights:float[]) (img:Mat) =
    detections |> Array.iteri (fun i d ->
        let clr = Scalar(0.,weights.[i] * weights.[i] * 200., 0.)
        Cv2.Rectangle(img,d,clr,5)
        printfn "weight %d %f" i weights.[i])

//Create a heat map from the given detections.
//It increments a counter for each pixel covered by a rectangle
let heatmapRects (sz:Size) (detections:Rect[]) =
    let plane = Array2D.create sz.Height sz.Width   0uy
    let ds = detections |> Array.filter (fun r->r.Width > 5 && r.Height > 5)
    for r in ds do
        for i in r.Y..r.Y+r.Height-1 do
            for j in r.X..r.X+r.Width-1 do
                let v = plane.[i,j]
                plane.[i,j] <- v + 1uy |> min 255uy
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

//kick off SVM training - it can take several minutes, depending on the number of images
let trainDetector() =
    printfn "start training"
    trainSvm (hog trainWinSize) posTrain negTrain svmFile
    printfn "done training"
   
//input and output video files paths
let v_prjctP = @"D:\repodata\obj_detect\project_video.mp4"
let v_prjctT = @"D:\repodata\obj_detect\test_video.mp4"
let v_prjctPOut = @"D:\repodata\obj_detect\project_video_out.mp4"
let v_prjctTOut = @"D:\repodata\obj_detect\test_video_out.mp4"

//process the input video file to detect objects
//and write an annotated video to the output
//(the annotations are bounding boxes over the detected objects)
let testVideoDetect (v_prjct:string) (v_out:string) =
    let detector = detect (Size(4.,4.)) 0.1 0 //configure the detection mehtod

    let hd = hog trainWinSize                 //instantiate HOGDescriptor
    let svmT = SVM.Load(svmFile)           //load SVM model from file
    setDetector hd svmT                       //configure HOGDescriptor with SVM

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
            let rcts,wts = detector hd n                       //run the frame through the HOGDescriptor to detect objects
            let rcts = rcts |> Array.filter (fun r->           //filter out detections in the sky or too close to the bottom
                    r.Bottom >= int (float imgSz.Height * 0.3) 
                    && r.Bottom <= int (float imgSz.Height * 0.95))
            let bbs = findBoundingBoxes 30. m rcts             //merge overlapping detections
            bbs |> Array.iter (fun b -> Cv2.Rectangle(m,b,Scalar(0.,255.,255.),2))  //draw bounding boxes over detected objects
            clipOut.Write(m)
            let fn = Path.Combine(folder,sprintf "th%d.jpg" !r)
            m.SaveImage(fn) |> ignore
            r := !r + 1
            printfn "th %d" !r
            m.Release()
            n.Release()
        with ex ->
            printfn "frame miss %d %s" !r ex.Message
    clipIn.Release()
    clipOut.Release()
;;

(*
testVideoDetect v_prjctP v_prjctPOut
testVideoDetect v_prjctT v_prjctTOut
launchSvmTraining()
testPrediction()
trainDetector()
*)

//Utility method to test the detector on a single image
let testDetector() =
    let padding = Size(4,4)
    let hd = hog trainWinSize
    let svmT = SVM.Load(svmFile)
    setDetector hd svmT
    let file,p = @"D:\repodata\adv_lane_find\imgs\img299.jpg",padding
    //let file,p = @"D:\repodata\obj_detect\test3\image_web_1.jpg",ep
    //let file,p = @"D:\repodata\obj_detect\vehicles\GTI_Right\image0199.png",zp
    let img = Cv2.ImRead(file)
    let normd = normalizeFrame img
    let wm = detect padding 0.2 0 hd normd
    let rects = fst wm
    let heatMap = heatmapRects (img.Size()) (fst wm)
    use heatMapGray = new Mat([img.Height;img.Width],MatType.CV_8UC1,heatMap)
    use heatMapTh = heatMapGray.EmptyClone()
    let cntrTh = 20.
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
    let rawRects = img.Clone()
    drawRects wm rawRects

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

//Utility methods to test the detector
//on images in a folder
let detectFile detector (hd:HOGDescriptor)  (file : string) = 
    let img = (Cv2.ImRead(file)) 
    use normd = normalizeFrame img
    let dtcts = detector hd normd
    ///drawRects dtcts img
    drawRectsSmpl dtcts img
    img
let testDctrOnImages() =
    let padding = Size(0,0)
    let folder = @"D:\repodata\obj_detect\test3"
    let negs = Directory.GetFiles(folder,"extra*.*")
    let poss = Directory.GetFiles(folder, "image*.*")
    let inImgs = Array.append poss negs
    let hd = hog trainWinSize
    let svmT = SVM.Load(svmFile)
    setDetector hd svmT
    let dtctr = detect padding 0.04 0
    let f = inImgs.[3]
    for f in inImgs do 
        use imgO = detectFile dtctr hd f
        imgO.SaveImage(Path.Combine(folder,"o_" + Path.GetFileName(f))) |> ignore
