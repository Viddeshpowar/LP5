#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void bfs(int node) 
{
	queue<int> q;
	q.push(node);
	visited[node] = true;
	while (!q.empty()) 
	{
		int curr_node = q.front();
		q.pop();
		cout << curr_node << " ";
		
		#pragma omp parallel for
		for (int i = 0; i < graph[curr_node].size(); i++) 
		{
			int adj_node = graph[curr_node][i];
			if (!visited[adj_node]) 
			{
				visited[adj_node] = true;
				q.push(adj_node);
			}
		}
	}
}

int main() {
    int n, m, start_node;
    cout << "Enter the number of nodes, edges, and the start node: ";
    cin >> n >> m >> start_node;

    cout << "Enter pairs of edges: ";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    bfs(start_node);

    return 0;
}
