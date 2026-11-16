# CLEVRER format report

## Checks

| status | check | detail |
|---|---|---|
| PASS | questions file present | 10000 videos, 152572 questions |
| PASS | every train/val question has a program | 47 distinct modules |
| PASS | annotation files found | 15000 files, e.g. train/annotation_train/annotation_00000-01000/annotation_00000.json |
| WARN | annotation locations are 3D world coords | no 2D boxes here; use derender proposals |
| PASS | collision events carry frame_id | 72 events in sample |
| PASS | object_property has color/material/shape |  |
| PASS | derender proposals found | 20000 files, e.g. derender_proposals/proposal_00000.json |
| PASS | proposal masks decode to 2D boxes | 50 decoded, sizes {(320, 480): 16793} |
| PASS | proposal objects carry attributes for matching to annotation objects |  |
| PASS | proposal frame count matches annotation frame count | {128: 30} |

## Questions: statistics

```
question types: {'descriptive': 109952, 'explanatory': 16799, 'counterfactual': 18642, 'predictive': 7179}

subtypes:
  counterfactual -                        18642
  descriptive    count                    40000
  descriptive    exist                    20000
  descriptive    query_color              18454
  descriptive    query_material           14614
  descriptive    query_shape              16884
  explanatory    -                        16799
  predictive     -                        7179

program present by type: {'descriptive': 109952, 'explanatory': 16799, 'counterfactual': 18642, 'predictive': 7179}
program MISSING by type: {}
distinct program modules: 47
top modules:
  unique                           558994
  objects                          456302
  filter_shape                     297627
  filter_color                     264815
  filter_collision                 264201
  events                           156291
  sphere                           103668
  filter_material                  103597
  cylinder                         102308
  all_events                       99043
  cube                             91651
  filter_in                        63933
  rubber                           51811
  metal                            51786
  belong_to                        42620
  count                            40000
  get_frame                        39701
  cyan                             34009
  brown                            33803
  gray                             33369
  purple                           33151
  green                            33102
  yellow                           32589
  red                              32550
  blue                             32242
  filter_moving                    25245
  filter_stationary                24707
  get_col_partner                  22301
  exist                            20000
  filter_order                     19502
  get_counterfact                  18642
  query_color                      18454
  negate                           17753
  query_shape                      16884
  filter_ancestor                  16799
  query_material                   14614
  end                              12606
  start                            11432
  null                             10251
  last                             9915

choices per MC question: {4: 22966, 3: 10188, 2: 9317, 1: 149}
choice answer values: {'wrong': 70375, 'correct': 70836}
correct choices per question (type, n_correct): {('explanatory', 2): 5541, ('explanatory', 3): 4320, ('counterfactual', 1): 7947, ('counterfactual', 2): 6406, ('explanatory', 1): 3428, ('predictive', 1): 7179, ('explanatory', 0): 2743, ('counterfactual', 3): 4120, ('explanatory', 4): 767, ('counterfactual', 0): 169}
descriptive answer vocab (21): [('1', 11079), ('yes', 10026), ('no', 9974), ('0', 9862), ('2', 9292), ('rubber', 7311), ('metal', 7303), ('3', 6525), ('sphere', 5648), ('cylinder', 5635), ('cube', 5601), ('4', 3173), ('gray', 2339), ('red', 2318), ('green', 2318), ('cyan', 2315), ('blue', 2301), ('yellow', 2299), ('brown', 2292), ('purple', 2272), ('5', 69)]
```

## Questions: sample descriptive

```
question_id: int = 0
question: str = 'What is the shape of the object to collide with the purple ...
question_type: str = 'descriptive'
question_subtype: str = 'query_shape'
program: list[13]
  [0]:
answer: str = 'sphere'

{
 "question_id": 0,
 "question": "What is the shape of the object to collide with the purple object?",
 "question_type": "descriptive",
 "question_subtype": "query_shape",
 "program": [
  "events",
  "objects",
  "purple",
  "filter_color",
  "unique",
  "filter_collision",
  "unique",
  "objects",
  "purple",
  "filter_color",
  "unique",
  "get_col_partner",
  "query_shape"
 ],
 "answer": "sphere"
}
```

## Questions: sample explanatory

```
question_id: int = 11
question: str = 'Which of the following is responsible for the collision bet...
question_type: str = 'explanatory'
program: list[15]
  [0]:
choices: list[4]
  [0]:
    choice_id: int = 0
    choice: str = 'the collision between the gray sphere and the purple sphere...
    program: list[16]
      [0]:
    answer: str = 'wrong'

{
 "question_id": 11,
 "question": "Which of the following is responsible for the collision between the gray object and the cube?",
 "question_type": "explanatory",
 "program": [
  "events",
  "events",
  "objects",
  "gray",
  "filter_color",
  "unique",
  "filter_collision",
  "objects",
  "cube",
  "filter_shape",
  "unique",
  "filter_collision",
  "unique",
  "filter_ancestor",
  "belong_to"
 ],
 "choices": [
  {
   "choice_id": 0,
   "choice": "the collision between the gray sphere and the purple sphere",
   "program": [
    "events",
    "objects",
    "gray",
    "filter_color",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "objects",
    "purple",
    "filter_color",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 1,
   "choice": "the presence of the metal sphere",
   "program": [
    "objects",
    "metal",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 2,
   "choice": "the blue rubber sphere's entering the scene",
   "program": [
    "events",
    "objects",
    "blue",
    "filter_color",
    "rubber",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique",
    "filter_in",
    "unique"
   ],
   "answer": "correct"
  },
  {
   "choice_id": 3,
   "choice": "the presence of the blue rubber sphere",
   "program": [
    "objects",
    "blue",
    "filter_color",

```

## Questions: sample counterfactual

```
question_id: int = 13
question: str = 'Which event will happen if the cylinder is removed?'
question_type: str = 'counterfactual'
program: list[7]
  [0]:
choices: list[4]
  [0]:
    choice_id: int = 0
    choice: str = 'The blue rubber sphere collides with the cube'
    program: list[16]
      [0]:
    answer: str = 'wrong'

{
 "question_id": 13,
 "question": "Which event will happen if the cylinder is removed?",
 "question_type": "counterfactual",
 "program": [
  "all_events",
  "objects",
  "cylinder",
  "filter_shape",
  "unique",
  "get_counterfact",
  "belong_to"
 ],
 "choices": [
  {
   "choice_id": 0,
   "choice": "The blue rubber sphere collides with the cube",
   "program": [
    "all_events",
    "objects",
    "blue",
    "filter_color",
    "rubber",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "objects",
    "cube",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 1,
   "choice": "The gray object collides with the metal sphere",
   "program": [
    "all_events",
    "objects",
    "gray",
    "filter_color",
    "unique",
    "filter_collision",
    "objects",
    "metal",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 2,
   "choice": "The cube and the metal sphere collide",
   "program": [
    "all_events",
    "objects",
    "cube",
    "filter_shape",
    "unique",
    "filter_collision",
    "objects",
    "metal",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 3,
   "choice": "The gray sphere and the cube collide",
   "program": [

```

## Questions: sample predictive

```
question_id: int = 13
question: str = 'What will happen next?'
question_type: str = 'predictive'
program: list[2]
  [0]:
choices: list[2]
  [0]:
    choice_id: int = 0
    choice: str = 'The cylinder and the rubber sphere collide'
    program: list[14]
      [0]:
    answer: str = 'wrong'

{
 "question_id": 13,
 "question": "What will happen next?",
 "question_type": "predictive",
 "program": [
  "unseen_events",
  "belong_to"
 ],
 "choices": [
  {
   "choice_id": 0,
   "choice": "The cylinder and the rubber sphere collide",
   "program": [
    "all_events",
    "objects",
    "cylinder",
    "filter_shape",
    "unique",
    "filter_collision",
    "objects",
    "rubber",
    "filter_material",
    "sphere",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "wrong"
  },
  {
   "choice_id": 1,
   "choice": "The cube and the cylinder collide",
   "program": [
    "all_events",
    "objects",
    "cube",
    "filter_shape",
    "unique",
    "filter_collision",
    "objects",
    "cylinder",
    "filter_shape",
    "unique",
    "filter_collision",
    "unique"
   ],
   "answer": "correct"
  }
 ]
}
```

## Annotations: statistics

```
sampled 30 of 15000 annotation files
top-level keys: {'scene_index': 30, 'video_filename': 30, 'object_property': 30, 'motion_trajectory': 30, 'collision': 30}
objects per video: {4: 12, 5: 12, 6: 6}
trajectory frames per video: {128: 30}
frame_id ranges seen: [(0, 127)]
per-frame fields: {'frame_id': 30, 'objects': 30}
per-object fields: {'object_id': 144, 'location': 144, 'orientation': 144, 'velocity': 144, 'angular_velocity': 144, 'inside_camera_view': 144}
location dimensionality: {3: 144}
inside_camera_view values: {True: 15680, False: 2752}
collisions per video: {2: 19, 3: 10, 4: 1}
collision fields: {'object_ids': 72, 'frame_id': 72, 'location': 72}
collision frame_id range: (15, 124)
```

## Annotations: structure of one file

```
scene_index: int = 13835
video_filename: str = 'video_13835.mp4'
object_property: list[4]
  [0]:
    object_id: int = 0
    color: str = 'gray'
    material: str = 'metal'
    shape: str = 'sphere'
motion_trajectory: list[128]
  [0]:
    frame_id: int = 0
    objects: list[4]
      [0]:
        object_id: int = 0
        location: list[3]
          ...
        orientation: list[3]
          ...
        velocity: list[3]
          ...
        angular_velocity: list[3]
          ...
        inside_camera_view: bool = True
collision: list[4]
  [0]:
    object_ids: list[2]
      [0]:
    frame_id: int = 32
    location: list[3]
      [0]:
```

## Proposals: statistics

```
sampled 30 of 20000 proposal files
top-level keys: {'video_index': 30, 'frames': 30, 'video_filename': 30}
frames per file: {128: 30}
per-frame keys: {'frame_index': 3840, 'objects': 3840}
objects per frame: {1: 31, 2: 117, 3: 644, 4: 1170, 5: 1352, 6: 526}
per-object keys: {'mask': 16793, 'video_index': 16793, 'frame_index': 16793, 'color': 16793, 'material': 16793, 'shape': 16793, 'score': 16793}
mask counts type: {'str': 16793}
mask sizes (h, w): {(320, 480): 16793}
RLE -> box decode: ok=50 fail=0, example normalized box=(0.6229166666666667, 0.490625, 0.7395833333333334, 0.703125)
annotation trajectory frames per video (for comparison): {128: 30}
```

## Proposals: structure of one file

```
video_index: int = 19723
frames: list[128]
  [0]:
    frame_index: int = 0
    objects: list[1]
      [0]:
        mask: dict[2]
          size: list[2]
            ...
          counts: str = 'Wdm2>]97WOi0K3M4N2M3O0N4N0000001N1O2O1O0O100000000000000000...
        video_index: int = 19723
        frame_index: int = 0
        color: str = 'brown'
        material: str = 'metal'
        shape: str = 'cylinder'
        score: float = 0.9996553659439087
video_filename: str = 'video_19723.mp4'
```

## Verdict: triplet field derivability

```
PASS  q_i sub-question from program        program templates per module
WARN  a_i intermediate answer              needs a program executor over annotations (port from chuangg/CLEVRER executor)
PASS  e_i box from proposals               mask -> tight box, normalized
PASS  e_i frame span                       collision frame_id; trajectory frame_id for object steps
PASS  d_i dependencies                     postfix program data flow
```
