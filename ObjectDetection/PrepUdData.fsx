#load "SetEnv.fsx"
open OpenCvSharp
open OpenCVCommon
open System
open System.IO
open SetEnv
open Utils
open OpenCvSharp.ML
open System.Runtime.InteropServices
open System.Threading
open System.Windows.Forms
open Probability
open FSharp.Data

//this script prepares images from different data sources
//for HOG based SVM training
//All images are sized to the 64x64 

let inFolder = @"D:\repodata\obj_detect\object-dataset"
let outFldrCars = @"D:\repodata\obj_detect\udcars"
type Lables = CsvProvider< @"D:\repodata\obj_detect\object-dataset\labels.csv">
let lbls = Lables.GetSample().Rows|>Seq.map(fun r->r.Label) |> Seq.distinct |> Seq.toArray
let cars = Lables.GetSample().Rows |> Seq.filter(fun r->r.Label="car" && r.Occluded=false)

let extractCars() =
    if Directory.Exists outFldrCars |> not then Directory.CreateDirectory outFldrCars |> ignore
    cars |> Seq.iteri(fun i r ->
        let file = Path.Combine(inFolder,r.Frame)
        use img = Cv2.ImRead(file)
        let roi = Rect(r.Xmin,r.Ymin,r.Xmax-r.Xmin,r.Ymax-r.Ymin)
        use p = img.SubMat(roi)
        use ps = new Mat()
        Cv2.Resize(!>p,!>ps,Size(64,64))
        let f = Path.Combine(outFldrCars, sprintf "i%i.png" i)
        ps.SaveImage(f) |> ignore
        )


let outFldrNonCars = @"D:\repodata\obj_detect\nonudcars"
let extractNonCars() =
    if Directory.Exists outFldrNonCars |> not then Directory.CreateDirectory outFldrNonCars |> ignore
    let rnd = XorshiftPRNG()
    cars |> Seq.iteri(fun i r ->
        let file = Path.Combine(inFolder,r.Frame)
        use img = Cv2.ImRead(file)
        let h = img.Height / 2
        let maxPy = img.Height - 64
        let maxPx = img.Width - 64
        let px = rnd.Next(0,maxPx)
        let py = rnd.Next(h,maxPy)
        let roi = Rect(px,py,64,64)
        use p = img.SubMat(roi)
        let f = Path.Combine(outFldrNonCars, sprintf "i%i.png" i)
        p.SaveImage(f) |> ignore
        )

let outFldrOther = @"D:\repodata\obj_detect\other"
let extractOther() =
    if Directory.Exists outFldrOther |> not then Directory.CreateDirectory outFldrOther |> ignore
    let rnd = XorshiftPRNG()
    cars |> Seq.iteri(fun i r ->
        let file = Path.Combine(inFolder,r.Frame)
        use img = Cv2.ImRead(file)
        let maxPy = float img.Height / 3.0 |> int // sky and trees
        let maxPx = img.Width - 64
        let px = rnd.Next(0,maxPx)
        let py = rnd.Next(0,maxPy)
        let roi = Rect(px,py,64,64)
        use p = img.SubMat(roi)
        let f = Path.Combine(outFldrOther, sprintf "i%i.png" i)
        p.SaveImage(f) |> ignore
        )

let wcOutFldr = @"D:\repodata\obj_detect\wc"
let extractWhiteCar() =
    if Directory.Exists wcOutFldr |> not then Directory.CreateDirectory wcOutFldr |> ignore
    use img = Cv2.ImRead(@"D:\repodata\obj_detect\wc_base.png")
    use p = new Mat()
    Cv2.Resize(!> img, !>p, Size(64,64))
    for i in 0..200 do
        let f = Path.Combine(wcOutFldr, sprintf "i%i.png" i)
        p.SaveImage(f) |> ignore
