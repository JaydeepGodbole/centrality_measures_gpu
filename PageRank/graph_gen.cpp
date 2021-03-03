#include <bits/stdc++.h>
using namespace std;
#define fastIO ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL)
typedef long long ll;

int main()
{
    fastIO;
    freopen("params.txt", "r", stdin);
    freopen("input.txt", "w", stdout);

    srand(time(0));

    int V,E,choice;
    cin >> V >> E >> choice;

    vector<pair<int,int>> edges;
    vector<set<int>> adj(V);

    if (choice) { // directed graph (assume outdegree!=0)
        cout << E << '\n';
        assert(E >= V);
        for (int u=0; u<V; u++) {
            int v = rand() % V;
            while (u == v) {
                v = rand() % V;
            }
            adj[u].insert(v);
            edges.push_back({u,v});
            E--;
        }
    }
    else { // undirected graph
        cout << 2*E << '\n';
    }

    while (E--) {
        int u = rand() % V;
        int v = rand() % V;
        if (u==v) {
            E++;
            continue;
        } 
        if (adj[u].find(v) == adj[u].end()) {
            adj[u].insert(v);
            edges.push_back({u,v});
            if (!choice) {
                adj[v].insert(u);
                edges.push_back({v,u});
            }
        }
        else {
            E++;
            continue;
        }
    }

    for (auto edge : edges) {
        cout << edge.first << " " << edge.second << '\n';
    }
    
}