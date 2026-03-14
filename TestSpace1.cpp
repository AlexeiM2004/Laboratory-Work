// Alexei Maiorov, Student ID: [11022726]
// Date: [08/02/2026]

#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

int string_experimentation()
{
  //Here is the guinea pig string
  std::string words = "\nWe are the minions";
  std::string part = words.substr(4,3);
  //The .substr(start_position, length) extracts part of a string to create a new variable
  std::cout << part;
  std::cout << words;
  for (int i=0; i < 10; i++){
    words += " Banana";
    std::cout << words;
    //Pretty dumb for loop but its good practice
  }
  size_t length = words.length();
  //The .length() gives a numerical length of the words string
  size_t size = words.size();
  //The .size() gives a numerical size
  //size() and length() generally do the same thing, but length is just for strings
  std::cout << "\n" << length << "\n" << size;
  return 0;
}

//This is a practically useful function that has applications for generating unique structured filenames, by taking usually un appendable class types,
//Adding a file ending, and classifying it then as a string
//Still a bit perplexed by this!

int string_stream_experiment()
{
  int file_index{123};
  //Creates a variabled, file_index, this is an integer "123"
  std::ostringstream output_stream;
  //This is a writing buffer, accepting various types for << use, its like the multi purpose glue
  output_stream << "FileData." << file_index;
  //The ostringsteam has allowed us to append file_index to Filedata. , which it wouldnt have been able to naturally
  std::string output_filename{output_stream.str()};
  //Now this creates a string variable called output_file name, that is the output_stream as a string class
  std::cout << output_filename << "\n";
  //This simply prints out the final file name
  output_stream.str("");
  return 0;
}

int arrays_vs_vect()
{
  const size_t n_a{7};
  //This creates a constant equal to 7
  //Uses the type size_t, this is generally best for arrays, vectors or large data types, as int/double can overflow, it also doesnt deal with negatives. It would
  //Simply just ignore the negative but carry through errors
  int a[n_a];
  //This creates an array called "a" that has n_a entries, i.e. 7. 
  //Might be useful for creating an undefined array where we can alter the number of entries. 
  for(size_t i{}; i<n_a; ++i) a[i] = static_cast<double>(i+1);
  //Seems like this uses a for loop to iterate all the entries of the array up to its 7th entry by assigning each entry a double type, 0+1,1+1,2+1,3+1,4+1,5+1 etc...
  //The use of static_cast<double> states that it converts the integer i+1 into a double.
  for(size_t i{}; i<n_a; ++i) std::cout<<"a[" << i << "] =" << a[i] << "\n";
  //Another for loop iterating up to all entries, it just prints out what the entries are and assigns them what is in each entry.
  return 0; 
}

//The aim of this function is to print out all the even numbers from 2 to 20

int arrays_vs_vectors_practice()
{
  const size_t array_length{20};
  //Defines the length of the array
  double even_number_array[array_length];
  for(size_t i{}; i < array_length; ++i)
  //Creates a for loop w/ dummy index i , which iterates until i is greater than the array length, by adding one each time
  {
    even_number_array[i] = static_cast<double>(i+1);
    //Assigns each entry of the arry in ascending order adding one each time, 0 is 1 , 1 is 2 etc...
  }
  for(size_t j = 1; j < array_length; j += 2)
  //This foor loop starts from index 1, and adds 2 each time to the index, 1,3,5 etc until it terminates at 20.
  {
  std::cout<<"[" << j << "] =" << even_number_array[j] << "\n";
  //The output is then printed by showing the each iteration 1,3,5.. with its corresponding even number number output,
  //Notice that all even outputs are assigned to an odd index at 0 corresponds to 1
  }

  return 0;
}

//The aim of this function is the play with arrays, understand how to use them, identify if they're an alternative to arrays
//Research states that most of the time a vector can be used in place of an array due to its ability to have entries manipulated, appended or removed
//Typically an array may be used over a vector for performance or optimisation

void arrays_vs_vectors()
{
  const size_t vector_length{5};
  std::vector<double> vector_example_one(vector_length);
  //So we have set up the length of the vector and created a vector with 5 empty entries.
  for (size_t i = {}; i < vector_length; ++i)
  {
    vector_example_one[i] = 2.5*static_cast<double>(i+1);
    std::cout << vector_example_one[i] << "\n";
  }
}

double spring_const_calculator()
{
  double pixels_to_metres = 0.089e-6;
  double temperature = (273.15+21);
  double boltzman_constant = 1.38064e-23;
  std::vector<double> x_vals = {0.82045, 0.9506, 0.91932, 1.02528, 1.24588, 1.46915, 4.25594};
  std::vector<double> y_vals = {0.77552, 0.82282, 0.97679, 1.16546, 1.20277, 1.67261, 4.99406};
  for(int i = 0; i < x_vals.size(); ++i)
  {
    double kappa_x = (boltzman_constant * temperature) / std::pow( (x_vals[i]) * pixels_to_metres , 2);
    std::cout << kappa_x << "\n";
  }
  for(int i = 0; i < x_vals.size(); ++i)
  {
    double kappa_y = (boltzman_constant * temperature) / std::pow( (y_vals[i]) * pixels_to_metres , 2);
    std::cout << kappa_y << "\n";
  }
  return 0;
}

int main()
{
  //string_experimentation();
  //string_stream_experiment();
  //arrays_vs_vectors();
  //arrays_vs_vectors_practice();
  //arrays_vs_vectors();
  spring_const_calculator();

  return 0;
}