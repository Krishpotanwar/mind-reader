# Presentation script

*Not yet checked, so do not claim: live eyes test (G2), pilot (G3), timed Colab run (G4).*

**0:00 Cell 3**

Hello, I'm Krish. This is Mind-Reader, a webcam rock-paper-scissors bot called CHANAKYA. My question: are people predictable? In a public 2014 study, 72 players played 300 rounds each. After losing a round, they threw the same move again 52.7% of the time. A random player would do that 33.3% of the time. That figure is computed from the study's public data. So people have habits, and a model can learn them.

**0:30 Cells 6, 10, 30, 23**

Part one is the eyes. MediaPipe, a pretrained Google model, only turns a picture into 21 points on the hand. We do not train it; our scikit-learn models learn from those points. This table is the data, one row per picture. Find-S, from Lab 3, learns scissors as index and middle up, ring and pinky down. With noisy thresholds, ring and pinky turn into question marks, so real data needs models that cope with noise. Here are six models on the 372 official test pictures. Four reach 99.2%, the ceiling, because three pictures have no hand found. Luck is 33.3%. The confusion matrix has every picture with a hand on the diagonal. But these hands are drawn by a computer. The real test of the eyes is the live game, with real hands.

**1:30 Cells 36, 39, 54**

Part two is the brain. It predicts your next move from the last two rounds. This chart is the habit: players repeat most after a loss. We test only on players the model has never seen: five folds, and no player on both sides. Here are five brains next to three yardsticks: luck at 33.3%, always-rock at 35.6%, and always-repeat at 44.0%. The best brain, a decision tree, scores 45.5%. The SVM ties it, because three simple features make only 27 situations. That is only a small step above always-repeat. So this is evidence of predictability on unseen players. It is not a promise that the bot beats you. The Perceptron does worst of the five brains: it only updates on its mistakes, so on noisy labels it never settles.

**2:30 Cell 58**

Now the live game. I need one volunteer. Take the seat, press I agree to start the camera, and type your name.

**2:40 Cell 58**

Eight rounds. Hold your hand up and throw when the screen says SHOOT. Before every countdown, CHANAKYA locks its move. Python keeps it, and the page only gets a round number, so nothing you throw can change it. Round one assumes rock, because 58.3% of first throws were rock, computed from the study's public data. Round two uses a Personal brain that learns only from you. From round three, the decision tree plays. Wins drain the health bar and ties do no damage; more wins takes the match. If the camera misreads you, press one, two or three to correct it.

**3:45 Cells 58, 63, 65**

Here is the match report: the mind-read moments, shown as k of n, so you can see how small eight rounds is. One match is luck. The evidence is the study: 52.7% repeat after a loss, over 7,148 rounds that followed a loss, computed from the study's public data. As for my friends, the pilot has no data yet, so I make no claim about them.

**4:15 Cells 51, 52**

If there is time: the early rounds. In rounds two to ten the Personal brain scores 40.0%, against 39.2% for always-repeat, 34.9% for always-rock and 33.3% for luck. The edge is tiny. Over a whole game the Personal brain gains 3.2 points per 100 rounds, but always-repeat gains 1.3, because players repeat more as a game goes on.

**4:45 Cell 66**

In one line: people are a little predictable. The brain reads that at 45.5%, against always-repeat 44.0%, always-rock 35.6% and luck 33.3%. The eyes score 99.2% on computer-drawn hands; real hands are what the live game shows. Thank you.

## Questions

**1. Does the bot cheat by looking at my throw?**
No. Before the countdown, CHANAKYA predicts from the earlier rounds and Python stores its move. The page only gets a round number, so nothing about your throw can change it. If the camera misses your hand, the same move stays locked. The move comes back only after your throw is counted.

**2. Always repeat already gets 44%, so why is the brain worth anything?**
Honestly, only a little. The brain scores 45.5%, about 1.5 points above always-repeat at 44.0%; always-rock gets 35.6% and luck 33.3%. I claim evidence of predictability on unseen players, not that the bot wins. A single eight-round match says little either way.

**3. Did you train the hand detector yourself?**
No. MediaPipe is a pretrained Google model that only finds 21 points on the hand. Our scikit-learn logistic regression, trained on computer-drawn hands, names the gesture: 99.2% on the official test pictures, where luck is 33.3%. Real hands are the live eyes test, which I have not run yet.
