/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 200;  // TODO: Set the number of particles
  
  std::default_random_engine gen; // random generator
  
  Particle Particle_Obj; 
  
  // Normal (Gaussian) distribution for x, Y and theta
  normal_distribution<double> dist_x(x, std[0]);
  
  normal_distribution<double> dist_y(y, std[1]);
  
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // For each particles Initialize all the member values obtained from GPS and to default
  for (int i = 0; i < num_particles; ++i) 
  {
    Particle_Obj.id    = i;
    Particle_Obj.x     = dist_x(gen);
    Particle_Obj.y     = dist_y(gen);
    Particle_Obj.theta = dist_theta(gen);
    Particle_Obj.weight= 1.0;
    
    particles.push_back(Particle_Obj);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  double prediction_x,prediction_y,prediction_theta;

  
  for(int i = 0 ; i < num_particles; i++)
  {
    std::default_random_engine gen;
    
    // Predict values from velocity and yawrate
    prediction_x 	   = particles.at(i).x     + ((velocity / yaw_rate) * (sin(particles.at(i).theta + yaw_rate * delta_t) - sin(particles.at(i).theta)));
    prediction_y       = particles.at(i).y     + ((velocity / yaw_rate) * (cos(particles.at(i).theta)                      - cos(particles.at(i).theta + yaw_rate * delta_t )));
    prediction_theta   = particles.at(i).theta + (yaw_rate * delta_t);
      
    // Normal distribution from Predicted values
  	normal_distribution<double> dist_x(prediction_x, std_pos[0]);
	normal_distribution<double> dist_y(prediction_y, std_pos[1]);
	normal_distribution<double> dist_theta(prediction_theta, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}