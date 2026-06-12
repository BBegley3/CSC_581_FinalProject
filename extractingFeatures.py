def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")

    distance = 0.0
    for p1, p2 in zip(point1, point2):
        distance += (p1 - p2) ** 2

    return distance ** 0.5

def eye_length_ratio(p8,p9,p10,p11,p12,p13):
    eye1=euclidean_distance(p9,p10)
    eye2=euclidean_distance(p11,p12)
    return (max(eye1,eye2))/euclidean_distance(p8,p13)

def eye_dist_ratio(p0,p1,p8,p13):
    dist_between_eyes=euclidean_distance(p0,p1)
    return dist_between_eyes/euclidean_distance(p8,p13)

def nose_ratio(p15,p16,p20,p21):
    nose_width=euclidean_distance(p15,p16)
    return nose_width/euclidean_distance(p20,p21)

def lip_size_ratio(p2,p3,p17,p18):
    lip_height=euclidean_distance(p2,p3)
    lip_width=euclidean_distance(p17,p18)
    return lip_height/lip_width

def lip_length_ratio(p2,p3,p20,p21):
    lip_length=euclidean_distance(p2,p3)
    return lip_length/euclidean_distance(p20,p21)

def eyebrow_length_ratio(p4,p5,p6,p7,p8,p13):
    eyebrow1=euclidean_distance(p4,p5)
    eyebrow2=euclidean_distance(p6,p7)
    return (max(eyebrow1,eyebrow2))/euclidean_distance(p8,p13)

def aggresive_ratio(p10,p19,p20,p21):
    return euclidean_distance(p10,p19)/euclidean_distance(p20,p21)