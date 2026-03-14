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
    const double temp_kelvin = (273.15 + 22.3); 

    std::ifstream file("C:\\Users\\lexma\\projects\\TestSpace\\day_1_spots_data.csv");
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
    std::vector<double> z_positions;
    std::vector<double>t_positions;
    std::vector<double> frame_positions;
    std::vector<double> radius_positions;
    //As of yet not imperative information but may come in useful later
    std::vector<double> quality;
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
    //for (int i = {}; i < x_positions.size() ; ++i)
    //{
        //std::cout << "[" << i << "] " << x_positions[i] << " & " << y_positions[i] << "\n";
    //}

    //This section will be used to calculate the viscosity of water. 

    //Microscope calibration values ||TO BE CHANGED|| 
    double number_of_pixels = 44;
    double pixel_size = 10e-6 / number_of_pixels;
    double frames_per_second = 34.8432;

    //Converting frames into time, 

    std::vector<double> time_seconds;
    for(int i = 0; i < frame_positions.size(); ++i)
    {
        double frame = frame_positions[i]; 
        double time = frame / frames_per_second;
        time_seconds.push_back(time);
    }

    // OLD CODE ONLY USED FOR SANITY CHECK OF DISPLACEMENT AND FOR TAU = 1 LAG TIME
    // ----------------------------------------------------------------------------------------------------------------

    //Calculate squared displacement between frames

    std::vector<double> dx_displacements;
    std::vector<double> dy_displacements;

    //This is just used to check that overall the displacements are approximately 0
    std::vector<double> squared_displacements;
    for(int i = 1; i < x_positions.size(); ++i)
    {
        //This for loop essentially iterates by taking dx and dy, which are equal to dx = xth - (x-1)th posns,
        //It then calculates the squared displacements arrays by dx^2 plus dy^2
        double dx = x_positions[i] - x_positions[i-1];
        double dy = y_positions[i] - y_positions[i-1];
        dx_displacements.push_back(dx);
        dy_displacements.push_back(dy);
        squared_displacements.push_back(std::pow(dx,2)+std::pow(dy,2));
    }

    //Calculate averages of regular displacement, sanity check, should be ~ 0
    double average_dx = 0.0;
    double average_dy = 0.0;
    //Initiaise both averages to 0
    for(int i = 0; i < dx_displacements.size(); ++i)
    {
        average_dx += dx_displacements[i];
        average_dy += dy_displacements[i];
    }
    
    average_dx /= dx_displacements.size();
    average_dy /= dy_displacements.size();

    std::cout << "\n" << "Average dx " << average_dx;
    std::cout << "\n" << "Average dx \n" << average_dy;

    //Calculate the MSD, mean squared displacement, <dr^2>
    double MSD_sum = 0.0;
    //Initialising the sum to 0, to be changed
    for(int i = 0; i < squared_displacements.size(); ++i)
    {
        MSD_sum += squared_displacements[i];
        //Here we calculated the mean squared displacement sum by, MSD = MSD + Squared Disp
    }

    double MSD_pixels_squared = MSD_sum / squared_displacements.size();
    //This is our mean squared displacement in pixels squared

    //This section will be used to calculate our time averaged mean squared displacement TAMSD
    
    //End TAMSD section

    // ----------------------------------------------------------------------------------------------------------------

    std::vector<double> tau_values;
    std::vector<double> corrected_MSD_values;
    //For simplicity we will calculate the time lag for 20 intervals
    double time_between_frames = 1.0 / frames_per_second;
    int number_of_intervals = 100;
    for(int lag = 1; lag <= number_of_intervals; ++lag)
    {
        double tau = lag * time_between_frames;

        double MSD_sum = 0.0;
        int count = 0;
        //Initialise the MSD sum and the count to 0 

        for(int i = 0; i + lag < x_positions.size(); ++i)
        {
            double dx = x_positions[i + lag] - x_positions[i];
            double dy = y_positions[i + lag] - y_positions[i];
            //Basically says that dx = x(at i + lag) - x(at i)
            MSD_sum += std::pow(dx,2) + std::pow(dy,2);
            count++;
        }
        double corrected_MSD = MSD_sum / count;
        tau_values.push_back(tau);
        corrected_MSD_values.push_back(corrected_MSD);
        std::cout << "\n" << tau_values[lag-1] << " " << corrected_MSD_values[lag-1] << "\n";

    }

    //Here we will calculate our diffusion coefficient in terms of pixels squared per second
    //Based on the equation D = <dr^2> / 4*tau

    double diffusion_coeff = MSD_pixels_squared / ( 4.0 * time_between_frames);

    //Converting the diffusion_coefficient from pixels squared per section into metres per second
    double diffusion_coeff_corrected = diffusion_coeff * std::pow(pixel_size,2);
    double particle_radius = 0.53e-6;

    double viscosity = (boltzman_const * temp_kelvin) / (6 * particle_radius * diffusion_coeff_corrected * M_PI);
    std::cout << "\nThe Viscosity in kg /m /s is " << viscosity;

    //This is going to print out the values for our MSD and our tau


    return 0;
}

