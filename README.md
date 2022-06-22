# Smart heating

The goal of this project is to develop a **thermal model** of the building and **fit** the parameters of this model based on the temperature measurements of the
rooms in the building, the set temperatures, the outside temperature and heating powers. In order to finaly  use **model predictive control** to minimize the temperature difference between the room temperature and the set temperature

<p align="center">
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/house.png?raw=true" />
  <br>
  <em style="text-align:center">The considered house</em>
</p>

## Steps

1. **Literature review**
   - Review two scientific publications on thermal models of buildings.
   - Summarize the approach and the model used to thermally characterise the building. 
   - Propose a simple model related to our case.
   - [Report](reports/1_Literature_review.pdf)
  
2. **Parameter estimation**
   - Develop a multi-zone model of the building.
   - Complete case where the parameters of your model must be estimated for each room.
   - The interactions between the rooms must therefore be considered and each room has a different temperature evolution as a function of the adjacent rooms, the outside temperature and its heating input power.
   - [Report](reports/2_Parameter_estimation.pdf)

3. **Probabilistic modeling**
   - Make your model probabilistic. Integrate uncertainty on
     - the model parameters
     - the transition between timesteps
     - the temperature measurements.
     - [Report](reports/3_Uncertainty.pdf)
   - Consolidation from previous milestone 
     - [Report](reports/4_Consolidation.pdf)
  
4. **Predictive control**
   - Formulate an optimization problem to control the boiler and radiator operations to minimize the difference between the room temperature and its setpoint temperature (model predictive control).
   - Implement your objective and constraints function using Pyomo.
   - Solve your optimization problem using an appropriate solver and analyze your results.
   - [Report](reports/5_Model_predictive_control.pdf)

## Results

<p align="center">
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/chB.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/chHugo.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/chTim.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/cuisine.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/SAM.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/SDB.png?raw=true" />
  <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/SDS.png?raw=true" />
   <img src="https://github.com/Julien-Gustin/Smart-heating/blob/master/figures/Energy.png?raw=true" />
  <br>
  <em style="text-align:center">Results from the simulator</em>
</p>

## Authors

Julien Gustin, Joachim Houyon and Romain Charles