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
  
  num_particles = 40;  
  
  std::default_random_engine gen; // random generator
  
  particles = vector<Particle>(num_particles); // Initialize size of particles vector
  
  // Normal (Gaussian) distribution for x, Y and theta
  normal_distribution<double> dist_x(x, std[0]);
  
  normal_distribution<double> dist_y(y, std[1]);
  
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // For each particles Initialize all the member values obtained from GPS and to default
  for (int i = 0; i < num_particles; i++) 
  {
    Particle Particle_Obj; 
    Particle_Obj.id    = i;
    Particle_Obj.x     = dist_x(gen);
    Particle_Obj.y     = dist_y(gen);
    Particle_Obj.theta = dist_theta(gen);
    Particle_Obj.weight= 1.0;
    
    particles[i] = Particle_Obj;
    weights.push_back(Particle_Obj.weight);
  }
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  
  double prediction_x,prediction_y,prediction_theta;

  
  for(int i = 0 ; i < num_particles; i++)
  {
    std::default_random_engine gen;
    
    // Predict values from velocity and yawrate
    prediction_x 	   = particles[i].x     + ((velocity / yaw_rate) * (sin(particles.at(i).theta + yaw_rate * delta_t) - sin(particles.at(i).theta)));
    prediction_y       = particles[i].y     + ((velocity / yaw_rate) * (cos(particles.at(i).theta)                      - cos(particles.at(i).theta + yaw_rate * delta_t )));
    prediction_theta   = particles[i].theta + (yaw_rate * delta_t);
      
    // Normal distribution from Predicted values
  	normal_distribution<double> dist_x(prediction_x, std_pos[0]);
	normal_distribution<double> dist_y(prediction_y, std_pos[1]);
	normal_distribution<double> dist_theta(prediction_theta, std_pos[2]);
    
    // Update randomly selected values from the normal distribution to the particle
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
   // Find the nearest neigbour of each observation from the predicted data
   for(LandmarkObs ObservationIterator : observations)
   {
    double SmallDistance = -1;
    double Distance = -1;

    for (LandmarkObs predIterator : predicted) 
    {
      Distance = dist(predIterator.x, predIterator.y, ObservationIterator.x, ObservationIterator.y);
      
      if (SmallDistance == -1 || Distance < SmallDistance) 
      {
        SmallDistance = Distance;
        ObservationIterator.id = predIterator.id;       
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    // Constant normalizer for Gaussian Distribution
    const double ScalarMultiplier = (1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]));
  
  	double WeightSum = 0.0;
  	double exponent  = 0.0;
  
  	for (int i = 0; i < num_particles; i++) 
    { 	
      vector<LandmarkObs> AllObs_Transformed;
      vector<LandmarkObs> Landmarks_valid;


      /* Transform the observations from car coordinates systems to map(world) coordinates 
        using homogeneous tranformation matrix consisting of rotation and traslation operations*/
      for (auto ObsIter = observations.begin(); ObsIter < observations.end(); ObsIter++) 
      {
        LandmarkObs Obs_Transformed;
        Obs_Transformed.id = ObsIter->id;
        Obs_Transformed.x = particles[i].x + (cos(particles[i].theta) * ObsIter->x) - (sin(particles[i].theta) * ObsIter->y);
        Obs_Transformed.y = particles[i].y + (cos(particles[i].theta) * ObsIter->x) + (sin(particles[i].theta) * ObsIter->y);                                               
        AllObs_Transformed.push_back(Obs_Transformed);

      }

      // Consider map landmarks that are within the senorrange as valid. Ignore all other map landmarks
      for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
      {
        Map::single_landmark_s Landmark_Instance = map_landmarks.landmark_list[k];
        if (dist(particles[i].x, particles[i].y, Landmark_Instance.x_f, Landmark_Instance.y_f) <= sensor_range) 
        {
          Landmarks_valid.push_back(LandmarkObs {Landmark_Instance.id_i,Landmark_Instance.x_f,Landmark_Instance.y_f});
        }
      }
	
      // Get the nearest neighbour prediction for each observation
      dataAssociation(Landmarks_valid, AllObs_Transformed);

      // Compute weights using Multivariate Gaussian Distribution between the observation and the predicted values which has association
      for(unsigned int m = 0; m < AllObs_Transformed.size(); m++)
      {
        for(unsigned int n = 0; n < Landmarks_valid.size(); n++)
        {
          if ( AllObs_Transformed[m].id == Landmarks_valid[n].id)
          {
            exponent = exp(-1.0 * ((pow((AllObs_Transformed[m].x - Landmarks_valid[n].x), 2)/(2.0 * (std_landmark[0]*std_landmark[0])))
                                 + (pow((AllObs_Transformed[m].y - Landmarks_valid[n].y), 2)/(2.0 * (std_landmark[1]*std_landmark[1])))));

            particles[i].weight *= ScalarMultiplier * exponent;
          }
        }
      }

      WeightSum += particles[i].weight;

    }

  // Normlaize weights to get sum of weights = 1.0
  for (int l = 0; l < num_particles; l++)
  {
    particles[l].weight /= WeightSum;
    weights[l] = particles[l].weight;
  }

}

void ParticleFilter::resample() {
  vector<Particle> Particles_Resampled;
  std::default_random_engine gen;
  double beta = 0.0;

  sort(weights.begin(), weights.end());
  double Max_Weight = weights[0];
  int index = 0;
  std::uniform_real_distribution<double> Random_Weight(0.0, 2*Max_Weight);
  std::uniform_int_distribution<int> Random_particle(0, num_particles - 1);
  
  // Resample the particles based on the weights
  for(int i = 0; i< num_particles; i++)
  {
    beta +=  Random_Weight(gen);
    index = Random_particle(gen);
    while (beta > weights[index]) 
    {
        beta -= weights[index];
        index = (index + 1) % num_particles;
	}
    Particles_Resampled.push_back(particles[index]);
  }
  
  particles = Particles_Resampled;
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