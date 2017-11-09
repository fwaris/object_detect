module NMS
//non maximum supression
(*
#load "SetEnv.fsx"
*)
open OpenCvSharp

let area (r:Rect) = r.Width * r.Height |> float

let overlap (r1:Rect) (r2:Rect) = 
    let intr = r1.Intersect r2 |> area
    let a1,a2 = area r1, area r2
    if a1 = intr || a2 = intr then //check for complete overlap
        1.0
    else
        let union = (area r1) + (area r2) - intr
        intr / union

//main Non Maximum Supression algorithm
//adapted from https://github.com/Nuzhny007/Non-Maximum-Suppression/blob/master/nms.h

let private nms' overlapTh ls =
    let rec innerLoop  acc rem r rs ls =
        match ls with
        | [] -> loop ((r,rs)::acc) (List.rev rem)
        | (n,ns)::rest ->               
            if overlap r n > overlapTh then 
                    innerLoop acc rem r rs rest  //supress rect
            else
                innerLoop acc ((n,ns)::rem) r rs rest

    and loop acc rem = 
        match rem with
        | []            -> acc
        | (r,s)::rest   -> innerLoop acc [] r s rest
    loop [] ls

let private nms2' overlapTh ls =
    let rec innerLoop  acc rem rL r rs ls =
        match ls with
        | [] -> loop ((rL,rs)::acc) (List.rev rem)
        | (n,ns)::rest ->               
            if overlap r n > overlapTh then 
                //if n.Height < r.Height && (rs - ns) < 0.2f then 
                if n.Height < (rL:Rect).Height then 
                    innerLoop acc rem n r ns rest  //prefer smaller rect
                    //innerLoop acc rem n rs rest 
                else
                    innerLoop acc rem rL r rs rest  
            else
                innerLoop acc ((n,ns)::rem) rL r rs rest

    and loop acc rem = 
        match rem with
        | []            -> acc
        | (r,s)::rest   -> innerLoop acc [] r r s rest
    loop [] ls


//nms API method
let nms overlapTh (detections:(Rect*float32)[]) =
    let sorted = detections |> Seq.sortByDescending snd |> Seq.toList
    let r1 = nms' overlapTh sorted
    let r2 = nms' overlapTh (r1 |> List.sortByDescending snd)
    r2 |> List.map fst



(*

let testing() =
   let r1  = [|Rect(912,61,34,34),0.71237123f|]
   nms 0.5 0.f 0 r1
   
   let r2s = 
        [|
            (Rect(822,48,107,107),0.999927f)
            (Rect(832,48,107,107),0.999778f)
            (Rect(1105,45,121,121),0.999661f)
            (Rect(842,48,107,107),0.999507f)
            (Rect(1115,45,121,121),0.999416f)
            (Rect(1102,48,107,107),0.998984f)
            (Rect(1095,45,121,121),0.998808f)
            (Rect(812,48,107,107),0.998444f)
            (Rect(1080,52,94,94),0.997839f)
            (Rect(1072,48,107,107),0.997309f)
            (Rect(1112,48,107,107),0.997269f)
            (Rect(1040,52,94,94),0.997077f)
            (Rect(852,48,107,107),0.997042f)
            (Rect(1032,48,107,107),0.995269f)
            (Rect(1065,45,121,121),0.992628f)
            (Rect(1090,52,94,94),0.992129f)
            (Rect(835,45,121,121),0.991983f)
            (Rect(1042,48,107,107),0.991649f)
            (Rect(825,45,121,121),0.991504f)
            (Rect(673,9,269,269),0.991241f)
            (Rect(1092,48,107,107),0.989020f)
            (Rect(830,52,94,94),0.988906f)
            (Rect(820,52,94,94),0.988104f)
            (Rect(862,48,107,107),0.988009f)
            (Rect(827,42,134,134),0.985873f)
            (Rect(1030,52,94,94),0.984870f)
            (Rect(1082,48,107,107),0.981981f)
            (Rect(860,52,94,94),0.981166f)
            (Rect(650,13,256,256),0.978462f)
            (Rect(1085,45,121,121),0.978201f)
            (Rect(802,48,107,107),0.977316f)
            (Rect(817,42,134,134),0.977064f)
            (Rect(1022,48,107,107),0.974028f)
            (Rect(1097,42,134,134),0.972187f)
            (Rect(1125,45,121,121),0.970718f)
            (Rect(1062,48,107,107),0.965594f)
            (Rect(1145,45,121,121),0.964915f)
            (Rect(810,52,94,94),0.964422f)
            (Rect(840,52,94,94),0.957094f)
            (Rect(1135,45,121,121),0.954838f)
            (Rect(1025,45,121,121),0.950686f)
            (Rect(1107,42,134,134),0.946394f)
            (Rect(815,45,121,121),0.940564f)
            (Rect(792,48,107,107),0.938609f)
            (Rect(845,45,121,121),0.929258f)
            (Rect(800,52,94,94),0.926320f)
            (Rect(668,16,242,242),0.925754f) //-
            (Rect(1035,45,121,121),0.924848f)
            (Rect(1075,45,121,121),0.916471f)
            (Rect(660,13,256,256),0.912982f)
            (Rect(1070,52,94,94),0.908711f)
            (Rect(1117,42,134,134),0.904677f)
            (Rect(1050,52,94,94),0.894630f)
            (Rect(1100,39,148,148),0.894356f)
            (Rect(850,52,94,94),0.892647f)
            (Rect(1110,39,148,148),0.879330f)
            (Rect(1020,52,94,94),0.863366f)
            (Rect(678,16,242,242),0.849317f)
            (Rect(782,48,107,107),0.844664f)
            (Rect(1052,48,107,107),0.839962f)
            (Rect(670,13,256,256),0.838939f)
            (Rect(1090,39,148,148),0.838252f)
            (Rect(680,13,256,256),0.837929f)
            (Rect(1012,48,107,107),0.814127f)
            (Rect(837,42,134,134),0.809396f)
            (Rect(805,45,121,121),0.797137f)
            (Rect(1055,45,121,121),0.796563f)
            (Rect(817,55,80,80),0.796229f) //-
            (Rect(790,52,94,94),0.769041f)
            (Rect(690,13,256,256),0.743278f)
            (Rect(827,55,80,80),0.725401f)
            (Rect(320,65,40,40),0.715059f)
            (Rect(658,16,242,242),0.709763f)
            (Rect(683,9,269,269),0.708643f)
            (Rect(698,16,242,242),0.707026f)
         |]
   nms 0.2  r2s
  // val rects : Rect [] =
  //[|(x:320 y:425 width:40 height:40); (x:1020 y:412 width:94 height:94);
  //  (x:1125 y:405 width:121 height:121); (x:668 y:376 width:242 height:242);
  //  (x:1080 y:412 width:94 height:94); (x:817 y:415 width:80 height:80)|]

   ()
*)