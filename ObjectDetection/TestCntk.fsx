#load "SetEnv.fsx"
open OpenCVCommon
open OpenCvSharp

//let img = Cv2.ImRead(@"D:\repodata\obj_detect\non-vehicles\nonudcars\i41.png")
let test1 = @"D:/repodata/obj_detect/nonudcars/i8.png"
let test2 = @"D:/repodata/obj_detect/vehicles/udcars/i7.png"
let test3 = @"D:\repodata\obj_detect\vehicles\wc\i10.png"

let mdl = DetectorSettings.loadModel()

let img = Cv2.ImRead(test3)
let n  = new Mat()
Cv2.CvtColor(!>img,!>n, ColorConversionCodes.BGR2YCrCb)
Detector.testDetect n mdl

