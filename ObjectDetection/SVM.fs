module SVM

//HOG-SVM based object detection 
//this was used previously - now using Convoutional Net

open OpenCvSharp
open OpenCVCommon
open System
open System.IO
open OpenCvSharp.ML
open System.Runtime.InteropServices
open Probability

//windows size over which the Histogram of Gradient (HOG) featuers are extracted
let trainWinSize = Size(64, 64)  

//SVM file - trained SVMs are stored here and used later in prediction
let svmFile = @"D:\repodata\obj_detect\veh_detect.yml" 
let boardSvmFile = @"D:\repodata\obj_detect\veh_detect_board.yml" 

//create OpenCV HOG Descriptor 
//see OpenCV documentation for details
let hog (sz:Size) =
    let hd = new HOGDescriptor()
    hd.WinSize <- sz
    hd.BlockSize <- Size(24,24)
    hd.BlockStride <- Size(8,8)
    hd.CellSize <- Size(8,8)
    //hd.BlockSize <- Size(24,24)
    //hd.BlockStride <- Size(8,8)
    //hd.CellSize <- Size(8,8)
    hd.Nbins <- 9
    //hd.NLevels <- 10 //default
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

//normalize and compute HOG features for the input image
let compDesc (hd:HOGDescriptor) (img:Mat) =
    use normd = normalizeFrame img
    //let padding = Size(4.,4.)
    //let pts = hd.Compute(normd, padding = !> padding)
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
let testPct = 0.9    

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
    svm.TermCriteria <- (TermCriteria(CriteriaType.Eps + CriteriaType.MaxIter,10000,0.001))
    svm.Gamma <- 1.
    svm.KernelType <- SVM.KernelTypes.Linear
    svm.Nu <- 0.8//0.1=>0.84; 0.9=>0.95; 0.8=>0.95
    svm.P <- 0.9
    svm.C <- 0.001
    //EpsR: c0.001,p=0.9=>0.85; c=0.001;P=0.1=>0.82
    //NuSvR: c=0.1;Nu=0.9=>0.83; c=0.9; Nu=0.9=>0.84; c=10;Nu=0.9=>0.825; c=0.001;Nu=0.001=>0.82
    //c=0.001;Nu=1=>0.81
   // https://stackoverflow.com/questions/3416522/what-is-the-opencv-svm-type-parameter
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
let detect (padding:Size) (stride:Size) threshold group (hd:HOGDescriptor)  (img:Mat) =
    let mutable weights : float[] = null
    let stride  : Nullable<Size> = !> stride
    let padding : Nullable<Size> = !> padding
    //hog.detectMultiScale(segment, found, 0.0, winStride, padding, 1.01, 0.1);
    let detections = hd.DetectMultiScale(img,
                            &weights, 
                            hitThreshold=threshold,
                            winStride=stride,
                            padding=padding,
                            scale=1.1,
                            groupThreshold = group)
    detections,weights 


//kick off SVM training - it can take several minutes, depending on the number of images
let trainDetector() =
    printfn "start training"
    trainSvm (hog trainWinSize) posTrain negTrain svmFile
    printfn "done training"

//alternate draw rectangle method  
let drawRectsSmpl (detections,weights:float[]) (img:Mat) =
    detections |> Array.iteri (fun i d ->
        let clr = Scalar(0.,weights.[i] * weights.[i] * 200., 0.)
        Cv2.Rectangle(img,d,clr,5)
        printfn "weight %d %f" i weights.[i])


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
    let stride = Size(8,8)
    let folder = @"D:\repodata\obj_detect\test3"
    let negs = Directory.GetFiles(folder,"extra*.*")
    let poss = Directory.GetFiles(folder, "image*.*")
    let inImgs = Array.append poss negs
    let hd = hog trainWinSize
    let svmT = SVM.Load(svmFile)
    setDetector hd svmT
    let dtctr = detect padding stride 0.04 0
    let f = inImgs.[3]
    for f in inImgs do 
        use imgO = detectFile dtctr hd f
        imgO.SaveImage(Path.Combine(folder,"o_" + Path.GetFileName(f))) |> ignore
