﻿module Utils
//utility methods for opencv 

open System.IO
open OpenCvSharp

let uiCtx = System.Threading.SynchronizationContext.Current

//show image in an opencv window (background thread)
let winAsync t i =
    async{
        do! Async.SwitchToContext uiCtx
        new Window((t:string), WindowMode.AutoSize,i) |> ignore 
        } 

let win t i = winAsync t i |> Async.Start

let append2Name f n =
    let fldr = Path.GetDirectoryName(f)
    let fn = Path.GetFileNameWithoutExtension(f)
    let xt = Path.GetExtension(f)
    let fn = fn + n + xt
    Path.Combine(fldr,fn)

let dumpImageProject v_file frameIdx = 
    use clipIn = new VideoCapture(v_file:string)
    let (w,h) = clipIn.FrameWidth,clipIn.FrameHeight
    let fldr = Path.GetDirectoryName(v_file) + "/imgs"
    if Directory.Exists fldr |> not then Directory.CreateDirectory fldr |> ignore
    let frames = clipIn |> Seq.unfold (fun x -> try if x.Grab() then Some(x.RetrieveMat(),x) else None with _ -> None)
    let frameIdx  = set frameIdx
    frames |> Seq.iteri (fun i fn ->
        if frameIdx.Contains i then
            let file = Path.Combine(fldr,sprintf "img%d.jpg" i)
            fn.SaveImage(file) |> ignore
        fn.Release()
    )
    clipIn.Release()

let dumpImageProjectAll v_file  = 
    use clipIn = new VideoCapture(v_file:string)
    let (w,h) = clipIn.FrameWidth,clipIn.FrameHeight
    let fldr = Path.GetDirectoryName(v_file) + "/imgs"
    if Directory.Exists fldr |> not then Directory.CreateDirectory fldr |> ignore
    let r = ref 0
    while clipIn.Grab() do
        let m = clipIn.RetrieveMat()
        let file = Path.Combine(fldr,sprintf "img%d.jpg" !r)
        m.SaveImage(file) |> ignore
        r := !r + 1
        m.Release()
    clipIn.Release()

let dmp2 (m:Mat) =
    for r in 0..m.Rows-1 do
        for c in 0..m.Cols-1 do
            let v = m.Get<Vec2i>(r,c)
            printfn "%d, %d = %A" r c (v.Item0, v.Item1)

let private dtbyte (m:Mat) =
    for r in 0..9 do
        for c in 0..9 do
            let v = m.Get<byte>(r,c)
            printfn "%d, %d = %A" r c v

let private dtbyte3 (m:Mat) =
    for r in 0..9 do
        for c in 0..9 do
            let v = m.Get<Vec3b>(r,c)
            printfn "%d, %d = %d %d %d" r c v.Item0 v.Item1 v.Item2

let private dtfloat32 (m:Mat) =
    for r in 0..9 do
        for c in 0..9 do
            let v = m.Get<float32>(r,c)
            printfn "%d, %d = %A" r c v

let dmp10 (m:Mat) =
    let mt = m.Type()   
    if mt = MatType.CV_32FC1 then dtfloat32 m
    elif mt = MatType.CV_8UC1 then dtbyte m
    elif mt = MatType.CV_8UC3 then dtbyte3 m
    else failwithf "case not handled %A" m.Type


let dmp1 (m:Mat) =
    for r in 0..m.Rows-1 do
        for c in 0..m.Cols-1 do
            let v = m.Get<uint8>(r,c)
            printfn "%d, %d = %d" r c v

let dmp1Uint (m:Mat) =
    for r in 0..m.Rows-1 do
        let v = m.Get<uint8>(r,0)
        printfn "%d = %d" r v
