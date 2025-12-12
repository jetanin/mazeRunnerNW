package com.nw.maze;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

// import org.springframework.CollectionUtils;

public class Main {

    private static final int directions[][] = { { -1, 0 }, { 0, 1 }, { 1, 0 }, { 0, -1 } };
    // private static final String FILE_NAME = "src/com/nw/maze/maze_101_101.txt";
    private static final String FILE_NAME = "m100_100.txt";
    private static final int DEFAULT_BLOCK_SIZE = 20;
    private static final int MIN_BLOCK_SIZE = 4;   // Smallest visible cell
    private static final int MAX_BLOCK_SIZE = 80;  // Cap so small mazes don't blow up

    MazeFrame frame;
    MazeData data;

    public void initFrame() {
        data = new MazeData(FILE_NAME);
        // collect maze files from current directory
        String[] mazeFiles = listMazeFiles();
        // compute block size so the maze fits in the user screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int screenW = (int) screen.getWidth();
        int screenH = (int) screen.getHeight();
        // leave some space for OS chrome / controls
        int maxSceneW = Math.max(300, screenW - 150);
        int maxSceneH = Math.max(200, screenH - 150);
        int blockSize = DEFAULT_BLOCK_SIZE;
        // try to keep DEFAULT_BLOCK_SIZE but shrink until maze fits
        while (blockSize > MIN_BLOCK_SIZE && (blockSize * data.M() > maxSceneW || blockSize * data.N() > maxSceneH)) {
            blockSize--;
        }
        if (blockSize < MIN_BLOCK_SIZE) blockSize = MIN_BLOCK_SIZE;
        if (blockSize > MAX_BLOCK_SIZE) blockSize = MAX_BLOCK_SIZE;

        int frameW = Math.min(blockSize * data.M(), maxSceneW);
        int frameH = Math.min(blockSize * data.N(), maxSceneH);
        frame = new MazeFrame("Maze Solver", frameW, frameH, mazeFiles);
        frame.setResizable(true);
        frame.setLocationRelativeTo(null);
        frame.setControlListener(new MazeFrame.ControlListener() {
            @Override
            public void onRunRequested(String algorithmName) {
                frame.setControlsEnabled(false);
                new Thread(() -> {
                    try {
                        runWithAlgorithm(algorithmName);
                    } finally {
                        // Re-enable controls on EDT after run completes
                        javax.swing.SwingUtilities.invokeLater(() -> frame.setControlsEnabled(true));
                    }
                }, "maze-runner").start();
            }
            @Override
            public void onResetRequested() { resetState(); }
            @Override
            public void onMazeSelected(String mazeFile) {
                // Load maze on EDT to avoid concurrency issues with renderer
                javax.swing.SwingUtilities.invokeLater(() -> reloadMaze(mazeFile));
            }
        });
        frame.setSelectedMaze(FILE_NAME);
        frame.render(data);
        // Wait for user to press Run; no auto-execution
    }

    private String[] listMazeFiles() {
        java.io.File dir = new java.io.File(".");
        java.io.File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".txt"));
        if (files == null) return new String[] { FILE_NAME };
        java.util.Arrays.sort(files, java.util.Comparator.comparing(java.io.File::getName));
        String[] names = new String[files.length];
        for (int i = 0; i < files.length; i++) names[i] = files[i].getName();
        return names;
    }

    private void reloadMaze(String fileName) {
        try {
            // Replace data with new maze, recompute frame size based on screen and maze size
            data = new MazeData(fileName);
            // compute block size & frame size same as init
            Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
            int screenW = (int) screen.getWidth();
            int screenH = (int) screen.getHeight();
            int maxSceneW = Math.max(300, screenW - 150);
            int maxSceneH = Math.max(200, screenH - 150);
            int blockSize = DEFAULT_BLOCK_SIZE;
            while (blockSize > MIN_BLOCK_SIZE && (blockSize * data.M() > maxSceneW || blockSize * data.N() > maxSceneH)) blockSize--;
            if (blockSize < MIN_BLOCK_SIZE) blockSize = MIN_BLOCK_SIZE;
            if (blockSize > MAX_BLOCK_SIZE) blockSize = MAX_BLOCK_SIZE;
            int frameW = Math.min(blockSize * data.M(), maxSceneW);
            int frameH = Math.min(blockSize * data.N(), maxSceneH);
            frame.setCanvasSize(frameW, frameH);
            frame.setTitle("Maze Solver - " + fileName);
            resetState();
            frame.render(data);
        } catch (Exception e) {
            System.err.println("Failed to load maze: " + fileName + " -> " + e.getMessage());
        }
    }

    private void resetState() {
        for (int i = 0; i < data.N(); i++) {
            for (int j = 0; j < data.M(); j++) {
                data.visited[i][j] = false;
                data.path[i][j] = false;
                data.result[i][j] = false;
            }
        }
        frame.setTitle("Maze Solver - reset");
        frame.render(data);
    }

    private void runWithAlgorithm(String algo) {
        // Reset state arrays
        for (int i = 0; i < data.N(); i++) {
            for (int j = 0; j < data.M(); j++) {
                data.visited[i][j] = false;
                data.path[i][j] = false;
                data.result[i][j] = false;
            }
        }

        switch (algo) {
            case "BFS":
                runBFS();
                return;
            case "A*":
                runAStar();
                return;
            case "Genetic":
                runGeneticStub();
                return;
            case "Dijkstra":
            default:
                runDijkstra();
        }
    }

    private void runDijkstra() {
        // Dijkstra's algorithm on grid with per-cell weights
        int rows = data.N();
        int cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) dist[i][j] = Integer.MAX_VALUE;
        }

        Node start = new Node(data.getEntranceX(), data.getEntranceY(), 0, null);
        if (data.inArea(start.x, start.y)) {
            dist[start.x][start.y] = 0;
        }

        PriorityQueue<Node> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.cost));
        pq.add(start);

        boolean isSolved = false;
        int visitedCount = 0;
        int visitedWeightSum = 0;
        long t0 = System.nanoTime();
        Node endNode = null;

        while (!pq.isEmpty()) {
            Node cur = pq.poll();
            if (data.visited[cur.x][cur.y]) continue; // finalized already
            data.visited[cur.x][cur.y] = true;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            visitedCount++;

            setData(cur.x, cur.y, true); // visualize exploration

            if (cur.x == data.getExitX() && cur.y == data.getExitY()) {
                isSolved = true;
                endNode = cur;
                break;
            }

            for (int[] d : directions) {
                int nx = cur.x + d[0];
                int ny = cur.y + d[1];
                if (!data.inArea(nx, ny)) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue; // skip walls
                if (data.visited[nx][ny]) continue;
                int stepCost = 1;
                if (data.weight != null) {
                    int w = data.weight[nx][ny];
                    stepCost = (w > 0 ? w : 1);
                }
                int newCost = (cur.cost == Integer.MAX_VALUE ? Integer.MAX_VALUE : cur.cost + stepCost);
                if (newCost < dist[nx][ny]) {
                    dist[nx][ny] = newCost;
                    pq.add(new Node(nx, ny, newCost, cur));
                }
            }
        }

        long t1 = System.nanoTime();

        if (isSolved && endNode != null) {
            int steps = findPath(endNode); // mark result path and count steps
            int totalCost = endNode.cost;
            long ms = (t1 - t0) / 1_000_000L;
            frame.updateMetrics(totalCost, steps, visitedCount, ms, "Dijkstra", visitedWeightSum);
        } else {
            frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "Dijkstra", visitedWeightSum);
            System.out.println("The maze has NO solution!");
        }
        setData(-1, -1, false);
    }

    private void runBFS() {
        ArrayDeque<Position> queue = new ArrayDeque<>();
        Position entrance = new Position(data.getEntranceX(), data.getEntranceY(), null);
        queue.add(entrance);
        if (data.inArea(entrance.x, entrance.y)) data.visited[entrance.x][entrance.y] = true;

        boolean isSolved = false;
        int visitedCount = 0;
        int visitedWeightSum = 0;
        long t0 = System.nanoTime();
        Position end = null;

        while (!queue.isEmpty()) {
            Position cur = queue.poll();
            visitedCount++;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            setData(cur.x, cur.y, true);
            if (cur.x == data.getExitX() && cur.y == data.getExitY()) { isSolved = true; end = cur; break; }
            for (int[] d : directions) {
                int nx = cur.x + d[0], ny = cur.y + d[1];
                if (data.inArea(nx, ny) && !data.visited[nx][ny] && data.getMazeChar(nx,ny)==MazeData.ROAD) {
                    data.visited[nx][ny] = true;
                    queue.add(new Position(nx, ny, cur));
                }
            }
        }

        long t1 = System.nanoTime();
        if (isSolved && end != null) {
            int steps = findPath(end);
            long ms = (t1 - t0) / 1_000_000L;
            frame.updateMetrics(null, steps, visitedCount, ms, "BFS", visitedWeightSum);
        } else {
            frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "BFS", visitedWeightSum);
        }
        setData(-1, -1, false);
    }

    private int findPath(Position p) {
        int steps = 0;
        Position cur = p;
        while (cur != null) {
            data.result[cur.x][cur.y] = true;
            cur = cur.prev;
            steps++;
        }
        return steps;
    }

    private void runAStar() {
        int rows = data.N(), cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i=0;i<rows;i++) for(int j=0;j<cols;j++) dist[i][j]=Integer.MAX_VALUE;
        Node start = new Node(data.getEntranceX(), data.getEntranceY(), 0, null);
        Node goal = new Node(data.getExitX(), data.getExitY(), 0, null);
        dist[start.x][start.y] = 0;

        Comparator<Node> cmp = (a,b) -> Integer.compare(a.cost + heuristic(a, goal), b.cost + heuristic(b, goal));
        PriorityQueue<Node> open = new PriorityQueue<>(cmp);
        open.add(start);

        boolean isSolved=false; int visitedCount=0; int visitedWeightSum=0; long t0=System.nanoTime(); Node end=null;
        while(!open.isEmpty()){
            Node cur = open.poll();
            if (data.visited[cur.x][cur.y]) continue;
            data.visited[cur.x][cur.y] = true; visitedCount++;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            setData(cur.x, cur.y, true);
            if (cur.x==goal.x && cur.y==goal.y){ isSolved=true; end=cur; break; }
            for(int[]d:directions){
                int nx=cur.x+d[0], ny=cur.y+d[1];
                if(!data.inArea(nx,ny) || data.getMazeChar(nx,ny)!=MazeData.ROAD || data.visited[nx][ny]) continue;
                int stepCost = data.weight!=null && data.weight[nx][ny]>0 ? data.weight[nx][ny] : 1;
                int newCost = cur.cost + stepCost;
                if(newCost < dist[nx][ny]){ dist[nx][ny]=newCost; open.add(new Node(nx,ny,newCost,cur)); }
            }
        }
        long t1=System.nanoTime();
        if(isSolved && end!=null){ int steps=findPath(end); long ms=(t1-t0)/1_000_000L; frame.updateMetrics(end.cost, steps, visitedCount, ms, "A*", visitedWeightSum); }
        else { frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "A*", visitedWeightSum); }
        setData(-1,-1,false);
    }

    private int heuristic(Node a, Node goal){
        // Manhattan distance as heuristic (weights ignored, admissible if weights>=1)
        return Math.abs(a.x - goal.x) + Math.abs(a.y - goal.y);
    }

    private void runGeneticStub() {
        // Placeholder for future GA: just show a message
        runGenetic();
    }

    private void runGenetic() {
        // Simple genetic algorithm: evolve sequences of moves with fitness = path cost to reach goal (penalize walls)
        final int maxGenerations = 200;
        final int populationSize = 200;
        // genome length will be dynamic based on maze size
        Random rnd = new Random(42);

        // Precompute distance map (BFS from goal) once for this run, used by the fitness function.
        final int[][] distMap = computeDistanceMap();

        // dynamic genome length bounds: min and max derived from maze size
        int maxGenomeLength = data.N() * data.M(); // upper bound
        int minGenomeLength = Math.max(10, Math.min(100, (data.N() + data.M()) * 2)); // reasonable lower bound
        if (minGenomeLength > maxGenomeLength) minGenomeLength = maxGenomeLength;

        // Helper to evaluate a genome (add fitness field)
        class EvalResult { int cost; int steps; int visited; int visitedWeight; List<int[]> path; boolean reached; int bfsDistance; double fitness; }
        Function<int[], EvalResult> evaluate = genome -> {
            // reset temp visited
            boolean[][] seen = new boolean[data.N()][data.M()];
            int x = data.getEntranceX(), y = data.getEntranceY();
            int cost = 0, steps = 0, visited = 0, visitedWeightSum = 0;
            ArrayList<int[]> path = new ArrayList<>();
            path.add(new int[]{x,y});
            seen[x][y] = true; visited++;
            int startW = data.weight!=null?data.weight[x][y]:1; visitedWeightSum += (startW>0?startW:1);
            for (int i=0;i<genome.length;i++) {
                int[] d = directions[genome[i]%4];
                int nx = x + d[0], ny = y + d[1];
                if (!data.inArea(nx, ny) || data.getMazeChar(nx,ny)!=MazeData.ROAD) {
                    cost += 5; // penalty for invalid move
                    continue;
                }
                x = nx; y = ny; steps++;
                int w = data.weight!=null?data.weight[x][y]:1; int stepCost = w>0?w:1; cost += stepCost;
                if (!seen[x][y]) { seen[x][y]=true; visited++; visitedWeightSum += stepCost; }
                path.add(new int[]{x,y});
                if (x==data.getExitX() && y==data.getExitY()) break;
            }
            EvalResult r = new EvalResult(); r.cost=cost; r.steps=steps; r.visited=visited; r.visitedWeight=visitedWeightSum; r.path=path; r.reached=(x==data.getExitX() && y==data.getExitY());
            // Use precomputed distance map; if out-of-area or unreachable, keep Integer.MAX_VALUE
            if (!data.inArea(x, y)) r.bfsDistance = Integer.MAX_VALUE;
            else r.bfsDistance = distMap != null ? distMap[x][y] : bfsDistance(x, y);
            // Compute fitness:
            // validRatioScore = (stepsValid / genomeLength) * 50.0
            // progressScore = ((distStart - distCurrent) / (double) distStart) * 50.0
            // if reached goal: fitness = 1000 + (genomeLength - stepsTaken)
            int sx = data.getEntranceX(), sy = data.getEntranceY();
            int distStart = (data.inArea(sx, sy) && distMap!=null) ? distMap[sx][sy] : Integer.MAX_VALUE;
            double validRatioScore = genome.length > 0 ? (steps / (double) genome.length) * 50.0 : 0.0;
            double progressScore = 0.0;
            if (distStart > 0 && distStart != Integer.MAX_VALUE) {
                double distCur = r.bfsDistance == Integer.MAX_VALUE ? distStart : r.bfsDistance;
                progressScore = ((distStart - distCur) / (double) distStart) * 50.0;
                if (progressScore < 0.0) progressScore = 0.0;
            }
            if (r.reached) {
                r.fitness = 1000.0 + (genome.length - r.steps);
            } else {
                r.fitness = validRatioScore + progressScore;
            }
             return r;
         };

        // Initialize population
        List<int[]> pop = new ArrayList<>();
        for (int i=0;i<populationSize;i++){
            int len = minGenomeLength + (maxGenomeLength==minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
            int[] g=new int[len];
            for(int j=0;j<len;j++) g[j]=rnd.nextInt(4);
            pop.add(g);
        }

        double bestScore = Double.NEGATIVE_INFINITY; Integer bestCost = null; int bestSteps=0, bestVisited=0, bestVisitedWeight=0; List<int[]> bestPath=null; String algoName="Genetic";
        long t0 = System.nanoTime();
        for (int gen=0; gen<maxGenerations; gen++) {
            // Evaluate
            List<EvalResult> results = new ArrayList<>(populationSize);
            for (int[] g : pop) results.add(evaluate.apply(g));
            // Select top 20% by fitness (descending)
            List<Integer> indices = new ArrayList<>();
            for (int i=0;i<results.size();i++) indices.add(i);
            indices.sort((a,b) -> Double.compare(results.get(b).fitness, results.get(a).fitness));
            List<int[]> next = new ArrayList<>();
            int eliteCount = Math.max(1, populationSize/5);
            for (int i=0;i<eliteCount;i++) next.add(pop.get(indices.get(i)));
            // Track best
            EvalResult br = results.get(indices.get(0));
            if (br.fitness > bestScore) { bestScore = br.fitness; bestSteps=br.steps; bestVisited=br.visited; bestVisitedWeight=br.visitedWeight; bestPath=br.path; if (br.reached) { if (bestCost==null || br.cost < bestCost) bestCost = br.cost; } }
            // Crossover + mutation to refill
            while (next.size() < populationSize) {
                int[] p1 = pop.get(rnd.nextInt(eliteCount));
                int[] p2 = pop.get(rnd.nextInt(eliteCount));
                // child length can vary; average + small noise or random within bounds
                int childLen = minGenomeLength + (maxGenomeLength==minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
                int[] child = new int[childLen];
                // pick cut points in parents
                int cut1 = rnd.nextInt(p1.length + 1);
                int cut2 = rnd.nextInt(p2.length + 1);
                // copy prefix from parent1
                int copyFromP1 = Math.min(cut1, childLen);
                if (copyFromP1 > 0) System.arraycopy(p1, 0, child, 0, copyFromP1);
                // fill remaining from p2 starting at cut2 with wrap
                int pos = copyFromP1;
                while (pos < childLen) {
                    child[pos] = p2[(cut2 + (pos - copyFromP1)) % p2.length];
                    pos++;
                }
                // mutation (gene replacement / insertion / deletion)
                int numMutations = Math.max(1, child.length / 20);
                for (int m=0;m<numMutations;m++) {
                    double op = rnd.nextDouble();
                    if (op < 0.80) {
                        // replace
                        child[rnd.nextInt(child.length)] = rnd.nextInt(4);
                    } else if (op < 0.90 && child.length < maxGenomeLength) {
                        // insert
                        int at = rnd.nextInt(child.length + 1);
                        int[] tmp = new int[child.length + 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        tmp[at] = rnd.nextInt(4);
                        System.arraycopy(child, at, tmp, at + 1, child.length - at);
                        child = tmp;
                    } else if (child.length > minGenomeLength) {
                        // delete
                        int at = rnd.nextInt(child.length);
                        int[] tmp = new int[child.length - 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        System.arraycopy(child, at + 1, tmp, at, child.length - at - 1);
                        child = tmp;
                    }
                }
                 next.add(child);
             }
             pop = next;
             // Occasionally update UI with best metrics
            if (gen % 10 == 0) frame.updateMetrics(bestCost, bestSteps, bestVisited, (System.nanoTime()-t0)/1_000_000L, algoName, bestVisitedWeight);
         }
         long t1 = System.nanoTime();
         // Render best path
         resetState();
         if (bestPath != null) {
             for (int[] cell : bestPath) {
                 setData(cell[0], cell[1], true);
             }
         }
        frame.updateMetrics(bestCost, bestSteps, bestVisited, (t1-t0)/1_000_000L, algoName, bestVisitedWeight);
         setData(-1, -1, false);
     }

    // BFS distance for fitness calculation
    private int bfsDistance(int sx, int sy) {
        if (!data.inArea(sx, sy)) return Integer.MAX_VALUE;
        int rows = data.N(), cols = data.M();
        boolean[][] seen = new boolean[rows][cols];
        ArrayDeque<int[]> q = new ArrayDeque<>();
        q.add(new int[]{sx, sy, 0});
        seen[sx][sy] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0], y = cur[1], d = cur[2];
            if (x == data.getExitX() && y == data.getExitY()) return d;
            for (int[] dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (!data.inArea(nx, ny)) continue;
                if (seen[nx][ny]) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                seen[nx][ny] = true;
                q.add(new int[]{nx, ny, d + 1});
            }
        }
        return Integer.MAX_VALUE;
    }

    private static class Position {
        int x, y; Position prev;
        Position(int x, int y, Position prev){ this.x=x; this.y=y; this.prev=prev; }
    }

    private int findPath(Node p) {
        int steps = 0;
        Node cur = p;
        while (cur != null) {
            data.result[cur.x][cur.y] = true;
            cur = cur.prev;
            steps++;
        }
        return steps;
    }

    private void setData(int x, int y, boolean isPath) {
        if (data.inArea(x, y)) {
            data.path[x][y] = isPath;
        }
        frame.render(data);
        MazeUtil.pause(frame.getDelayMs());
    }

    // No Position class needed after switching to Dijkstra's Node representation

    private class Node {
        private int x, y;
        private int cost;
        private Node prev;

        private Node(int x, int y, int cost, Node prev) {
            this.x = x;
            this.y = y;
            this.cost = cost;
            this.prev = prev;
        }
    }

    // Compute distance-from-exit map with BFS (distance in steps, Integer.MAX_VALUE = unreachable).
    private int[][] computeDistanceMap() {
        int rows = data.N(), cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) dist[i][j] = Integer.MAX_VALUE;
        int ex = data.getExitX(), ey = data.getExitY();
        if (!data.inArea(ex, ey)) return dist;
        ArrayDeque<int[]> q = new ArrayDeque<>();
        dist[ex][ey] = 0;
        q.add(new int[]{ex, ey});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0], y = cur[1], d = dist[x][y];
            for (int[] dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (!data.inArea(nx, ny)) continue;
                if (dist[nx][ny] != Integer.MAX_VALUE) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                dist[nx][ny] = d + 1;
                q.add(new int[]{nx, ny});
            }
        }
        return dist;
    }

    public static void main(String[] args) {
        new Main().initFrame();
    }

}
