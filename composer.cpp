// composer.cpp, Evan DeLord
// Description: Build the source code
// for composer.h, written by Maggie Johnson.
// Records info about composers.

#include <iostream> 
#include <string>
#include "composer.h"

using namespace std;

//Constructor
Composer::Composer()
{

	first_name_ = last_name_ = composer_genre_ = fact_ = "unspecified";
	composer_yob_ = -1;
	ranking_ = kDefaultRanking;
}
//Destructor
Composer::~Composer()
{
	first_name_.clear(); last_name_.clear(); composer_genre_.clear(); 
	fact_.clear(); 
}
//Accessors and Mutators
void Composer::set_first_name(string in_first_name)
{
	first_name_ = in_first_name;
}

string Composer::first_name()
{
	return first_name_;
}

void Composer::set_last_name(string in_last_name)
{
	last_name_ = in_last_name;
}

string Composer::last_name()
{
	return last_name_;
}

void Composer::set_composer_yob(int in_composer_yob)
{
	composer_yob_ = in_composer_yob;
}

int Composer::composer_yob()
{
	return composer_yob_;
}

void Composer::set_composer_genre(string in_composer_genre)
{
	composer_genre_ = in_composer_genre;
}

string Composer::composer_genre()
{
	return composer_genre_;
}

void Composer::set_ranking(int in_ranking)
{
	ranking_ = in_ranking;
}

int Composer::ranking()
{
	return ranking_;
}

void Composer::set_fact(string in_fact)
{
	fact_ = in_fact;
}

string Composer::fact()
{
	return fact_;
}

//Methods
// This method increases a composer's rank by increment.
void Composer::Promote(int increment)
{
	int temp_rank = ranking_ - increment;
	if(temp_rank < 1)
		ranking_ = 1;
	else
		ranking_ = temp_rank;
}
// This method decreases a composer's rank by decrement.
void Composer::Demote(int decrement)
{
	int temp_rank = ranking_ + decrement;
	if(temp_rank > kDefaultRanking)
		ranking_ = kDefaultRanking;
	else
		ranking_ = temp_rank;
}
// This method displays all the attributes of a composer.
void Composer::Display()
{
	cout << endl << "******" << endl;
	cout << "Name: " << first_name() << " " << last_name() << endl;
	cout << "Year of Birth: " << composer_yob() << endl;
	cout << "Genre: " << composer_genre() << endl;
	cout << "Ranking: " << ranking() << endl;
	cout << "Fact: " << fact() << endl;
	cout << "******" << endl;
}

