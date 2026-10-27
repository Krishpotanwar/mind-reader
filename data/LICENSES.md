# Data licences

The repo's code is MIT (see [LICENSE](../LICENSE)). The data files keep the licences of their sources.

## `rps_human_games.csv`

- **Source data:** Wang, Z., Xu, B. & Zhou, H.-J. (2014). *Social cycling and conditional responses in the
  Rock-Paper-Scissors game.* Scientific Reports 4, 5830. <https://doi.org/10.1038/srep05830> (72 players, 21,600 rounds).
- **Distribution:** the `data.WXZ2014` table in the R package **stratEst 1.1.8** by Fabian Dvorak, from CRAN
  (<https://cran.r-project.org/package=stratEst>), licensed GPL-3. Its `COPYRIGHT.txt` credits the data to
  Zhijian Wang, Bin Xu and Hai-Jun Zhou.
- **Changes:** converted from R's `.rda` format to CSV by `tools/rda_to_csv.py`, and the `input` column (it only
  repeats the player's previous choice) is dropped.
- **Notice:** this CSV is a converted copy of data distributed in a GPL-3 package, so it is provided under the
  GNU General Public License version 3 (<https://www.gnu.org/licenses/gpl-3.0.html>), without any warranty.
  The MIT licence of the repo's code does not cover it.

## `hand_landmarks.csv`

- **Source images:** "Rock Paper Scissors Dataset" by Laurence Moroney (<https://laurencemoroney.com/datasets.html>),
  licensed CC BY 2.0 (<https://creativecommons.org/licenses/by/2.0/>). CGI hands on a white background:
  2,520 training images and 372 test images.
- **Changes:** the CSV holds numbers only, no pixels. For each image where a hand is found it keeps the file name,
  split, label, image size and the 21 hand landmark positions computed by Google's MediaPipe HandLandmarker model.
  Images without a detected hand have no row.
