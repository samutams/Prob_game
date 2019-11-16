# Prob_game

The project was inspired by a typical risk preference measuring game, the bomb risk elicitation task (BRET). 

My aim was to acquire knowledge how to make complex scripts in python and to learn more about the usage of neural networks (nn). In line with this I wanted to analyze the performance of neural network models on a simple game like BRET. In my project I created an implementation of evolutionary neural networks that finds the best nn based on evolutionary algorithm. 

My version of the game assumes that the player can chose from „n” boxes. In one of the boxes a bomb was placed. The player chooses the box by selecting a number between 1 and “n”. If the selected number is less than the bomb’s number than the player wins and get a payment equals to the selected number. Otherwise the bomb explodes, and the player’s payment is zero. 

My aim was to determine the optimal strategy for the box selection that provides the maximal payment if the game played in an iterative fashion. 

In the project I created several classes with certain features that provides great flexibility to set up the game, examine the theoretical optimum and optimize it using machine learning techniques. 

Main features: 
-	The place of the bomb can be fixed during the iterations or it can be assigned by choosing different probability density functions (binom, skewnorm, alpha, gamma…) – prob.py
-	The main characteristics of the chosen probability density function can also be examined, since a class was created to provide statistics and visualization about them. – prob_stats.py
-	The payment from an individual guess can be examined, furthermore the payment from an iterative game can also be assessed. – risky_bomb.py
-	A class was also created to examine the properties of the iterative games. – risky_bomb_stats.py
-	The theoretical optimal guess from the player can be easily determined if the probability density function - that is used to place the bomb - is known. This can be calculated by multiplying the probability values with the different possible payments (numbers between 1 and n) and then by finding the place with the maximum expected payment. The optimal guessing strategy can be determined also be neural networks (where evolutionary algorithm is used to determine the best nn). 
-	Other modification of the game is also implemented, where the player knows that the probability density function – used for the placement of the bomb – shall be 1 of k known functions. Another nn algorithm is used to find the which of these functions are actually used during the iteration of the game (Classification problem). This case the optimal strategy (determined by the evolutionary nn) is played against the not know probability density function based on the classification results. 
