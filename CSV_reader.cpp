// Alexei Maiorov, Student ID: [11022726]
// Date: [09/02/2026]

#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>
#include <fstream>

int main()
{
    //Constants 
    const double boltzman_const = 1.380649e-23;
    const double temp_kelvin = (273.15 + 22.4); 

    std::ifstream file("C:\\Users\\lexma\\projects\\TestSpace\\OIL_yeast_brownian_motion_2.csv");
    std::string data;

    //First we must read the first four "metadata" lines and then discard them.
    //This can be done by creating a for loop to iterate this 4 times
    for(int i = 0; i < 4; ++i)
    {
        std::getline(file,data);
    }
    //Now we are creating a vector to store the data we are going to use
    std::vector<double> x_positions;
    std::vector<double> y_positions;
    std::vector<double> z_positions; //Not to be used
    std::vector<double>t_positions;
    std::vector<double> frame_positions;
    std::vector<double> radius_positions;
    //As of yet not imperative information but may come in useful later
    std::vector<double> quality; //Not to be used
    std::vector<double> intensity;
    
    //This is our vector, that will contain our x positions
    while(std::getline(file,data))
    {
        std::stringstream ss(data);
        //This creates a string stream from the line, it turns the string into a "fake file"
        std::string token;
        //This is a dummy variable, that will temporarily hold our data
        for(int i = 0; i < 4; ++i)
        {
            std::getline(ss, token, ',');
            //This essentially states it reads until it reaches a comma, 4 times, so we are reading after the 4th comma (The 4th column)
        }
        std::string x_data,y_data,z_data,t_data,frame_data,radius_data;
        //This sets up a series of strings that hold the x, y, z, time, frame and radius data 
        if(std::getline(ss, x_data, ',') && std::getline(ss, y_data, ',') && std::getline(ss, z_data, ',') && std::getline(ss, t_data, ',') && std::getline(ss, frame_data, ',') && std::getline(ss, radius_data, ','))
        {
            try
            {
                double x = std::stod(x_data);
                double y = std::stod(y_data);
                double z = std::stod(z_data);
                double t = std::stod(t_data);
                double f = std::stod(frame_data);
                double r = std::stod(radius_data);
                //This converts the string from the file into a double
                x_positions.push_back(x);
                y_positions.push_back(y);
                z_positions.push_back(z);
                t_positions.push_back(t);
                frame_positions.push_back(f);
                radius_positions.push_back(r);
                //This adds the x data into our vector, as its a vector it grows to make room for this
            } catch(...) {   
            //Little failsafe to ensure that if the conversion fails, like if the string isnt a number, then it just skips the line
            }
        }
    }

    //This section displays what x positions we have collected

    std::cout << "Successfully read " << x_positions.size() << " x positions\n";
    std::cout << "Successfully read " << y_positions.size() << " y positions\n";
    std::cout << "Successfully read " << t_positions.size() << " t positions\n";
    std::cout << "Successfully read " << frame_positions.size() << " frame positions\n";
    std::cout << "Successfully read " << radius_positions.size() << " r positions\n";

    //This section will be used to calculate the viscosity of water. 

    //Microscope calibration values ||TO BE CHANGED|| 
    double number_of_pixels = 112;
    double pixel_size = 10e-6 / number_of_pixels;
    double frames_per_second = 34.8;

    //Converting frames into time, 

    std::vector<double> time_seconds;
    for(int i = 0; i < frame_positions.size(); ++i)
    {
        double frame = frame_positions[i]; 
        double time = frame / frames_per_second;
        time_seconds.push_back(time);
    }

    //Calculates the TAMSD

    std::vector<double> tau_values; // Double vector to store time lag values (Tau)
    std::vector<double> corrected_MSD_values; // Double vector to store the TAMSD values for each corresponding tau

    //For simplicity we will calculate the time lag for 50 intervals, this can be cut down if necessary on ORIGIN
    
    double time_between_frames = 1.0 / frames_per_second; // Time between consecustive frames 1 / FPS
    int number_of_intervals = 150; // Number of lag time intervals to calculate

    for(int lag = 1; lag <= number_of_intervals; ++lag) // Loop over each time lag
    {
        double tau = lag * time_between_frames; // Converts lag index to time (lag * time)

        double MSD_sum = 0.0; // Accumulator for squared displacements
        int count = 0; //Counts the number of displacment pairs

        //Initialise the TAMSD sum and the count to 0 

        // For current lag value, loops through all possible starting positions, i
        for(int i = 0; i + lag < x_positions.size(); ++i) 
        {
            // C
            double dx = x_positions[i + lag] - x_positions[i]; // Displacement in x
            double dy = y_positions[i + lag] - y_positions[i]; // Displacement in y
            //Basically says that dx = x(at i + lag) - x(at i)
            // Add squared displacmement to sum
            MSD_sum += std::pow(dx,2) + std::pow(dy,2);
            count++; //Adds one to the increment counter
        }
        // Averages over all possible starting positions, (This is the Time averaged mean squared displacement)
        double corrected_MSD = MSD_sum / count;
        tau_values.push_back(tau); // Stores the time lag (Tau)
        corrected_MSD_values.push_back(corrected_MSD); // Stores the associated TAMSD value
    }

    //Write our MSD(tau) and tau results into a new file called MSD results output

    std::ofstream outfile("C:\\Users\\lexma\\projects\\TestSpace\\OIL_yeast_brownian_motion_2.txt");
    for(int i = 0; i < tau_values.size(); ++i)
    {
        outfile << tau_values[i] << "," << corrected_MSD_values[i] * std::pow(pixel_size,2) << "\n";
    }
    outfile.close();

    //Here we will ask the user what their gradient from the ORIGIN plot was.
    double gradient;
    std::cout << "Please enter your ORIGIN gradient; ";
    std::cin >> gradient;

    double viscosity = (boltzman_const * temp_kelvin) / (6 * 0.53e-6 * (gradient/4) * M_PI);
    std::cout << "\nThe Viscosity in kg /m /s is " << viscosity;

    return 0;
}

