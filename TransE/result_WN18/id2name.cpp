#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <utility>

using namespace std;

map<string, string> id2name;

string ReplaceAll(std::string &str, const std::string& from, const std::string& to){
    size_t start_pos = 0; //string처음부터 검사
    while((start_pos = str.find(from, start_pos)) != std::string::npos)  //from을 찾을 수 없을 때까지
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // 중복검사를 피하고 from.length() > to.length()인 경우를 위해서
    }
    return str;
}


int main(){
    ifstream fin1("entity2vec_epoch_999.txt");
    ifstream fin2("id2name.txt");
    ofstream fout("name2vec.txt");

    string line;  
    while( getline(fin2, line) ){
        string id, name;
        id = line.substr(1, 8);
        name = line.substr(10);
        ReplaceAll( name, " ", "_");

        cout << id << " : " << name << endl;
        id2name[id] = name;
    }

    while( getline(fin1, line) ){
        string id = line.substr(0, 8);
        string vec = line.substr(8);

        string name = id2name[id];

        if( name == "" ){
            name = "NoName";
        }

        cout << "[" << id << "] := " << name << endl;
        fout << name << vec << endl;
    }

    fin1.close();
    fin2.close();
    fout.close();
    return 0;
}
