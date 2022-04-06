#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

long int hashing_value( long double x,  long int digits );

int main(int argc, char **argv) {
  string input,output,cmd;
  long double pre,fit,temp;
  long int D, run;
  ifstream fin;
  ofstream fout;
  //cout << argc;
  if( argc != 4 ) {
      cout << "./Hashing <input> <output> <digits>\n";
      cout << "./Hashing raw_data.txt hashing_data.txt 4 \n";
      return 1;
  }
  //cout << argv[1];
  //cout << argv[2];
  input = argv[ 1 ];
  output = argv[ 2 ];
  pre = atoi( argv[3] ); //number of decimal digits as in left of polong int
  fin.open( input, ios::in );
  fout.open( output, ios::out );
  if( !fin.is_open() ) {
    cout << "File: " << input << " not found!!!\n";
    return 1;
  }
  fin >> cmd;
  cout << cmd ;
  fout << setfill(' ') << setw(4) << cmd << "\t";
  fin >> cmd;
  fout << setfill(' ') << setw(11) << "Fitness" << "\t";
  fin >> cmd >> D;
  fout << setfill(' ') << setw(8*D) << "Solution" << "\t";
  fin >> cmd;
  fout << setfill(' ') << setw(11) << "Fitness" << "\t";
  fin >> cmd;
  fout << setfill(' ') << setw(8*D) << "Next_Solution" << "\n";
  while( true ) {
    fin >> run;
    if( fin.eof() )
      break;
    fin >> fit;
   
    fout << setfill(' ') << setw(4) << std::dec << run << "\t" << setw(11) << dec << hashing_value( fit, pre ) << "\t";
    for( long int i = 0; i < D; i++ ) {
      fin >> temp;

      cout << hex << hashing_value(temp,pre);
      fout << std::setfill ('0') << setw(8) << hex << hashing_value( temp, pre );
    }
    fin >> fit;
    fout << "\t" << setfill(' ') << setw(11) << dec << hashing_value( fit, pre ) << "\t";
    for( long int i = 0; i < D; i++ ) {
      fin >> temp;
      fout << setfill('0') << setw(8) << hex << hashing_value( temp, pre );
    }
    fout << "\n";
  }
  fin.close();
  fout.close();
  cout << " " << input << " was loaded!!!\n";
  cout << " " << output << " is ready!!!\n";

  return 0;
}

long int hashing_value( long double x, long int digits ) {
  long double a;
  long double prec;
  long int b;

  prec = pow( 10.0, -digits );
  a = x / prec;
  a = round( a );
  
  // if( a > 2147483647 ) {
  //   cout << " warning (hashing_value): out_of_range" << endl;
  //   return 2147483647;
  // }
  // if( a < -2147483648 ) {
  //   cout << " warning (hasshin_value): out_of_range" << endl;
  //   return -2147483648 ;
  // }
  b = (long int)a;

  return b;
}
