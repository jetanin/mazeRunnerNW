package com.nw.maze;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JSlider;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

public class MazeFrame  extends JFrame{
    
	private int canvasWidth;
	private int canvasHeight;
	private MazeCanvas canvas;
	private boolean fullscreen = false;
	private java.awt.Rectangle windowedBounds = null;
	
	private MazeData data;

	// Controls
	private JComboBox<String> algorithmBox;
	private JComboBox<String> mazeBox;
	private JButton runButton;
	private JButton resetButton;
	private JButton fullscreenButton;
	private JSlider speedSlider;
	private ControlListener controlListener;
	private String[] mazeFiles;
	// Metrics labels
	private JLabel costLabel;
	private JLabel stepsLabel;
	private JLabel visitedLabel;
	private JLabel timeLabel;
	private JLabel visitedWeightLabel;
	
	public MazeFrame(String title, int canvasWidth, int canvasHeight, String[] mazeFiles) {
		super(title);
		this.canvasWidth = canvasWidth;
		this.canvasHeight = canvasHeight;
		this.mazeFiles = mazeFiles;

		// Build UI
		this.canvas = new MazeCanvas();
		JPanel root = new JPanel(new BorderLayout());
		root.add(buildControlPanel(), BorderLayout.NORTH);
		root.add(this.canvas, BorderLayout.CENTER);
		this.setContentPane(root);
        
		this.pack();
		this.setResizable(true);
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		// Add keyboard shortcut for fullscreen (F11)
		javax.swing.InputMap im = root.getInputMap(javax.swing.JComponent.WHEN_IN_FOCUSED_WINDOW);
		javax.swing.ActionMap am = root.getActionMap();
		im.put(javax.swing.KeyStroke.getKeyStroke("F11"), "toggleFullscreen");
		am.put("toggleFullscreen", new javax.swing.AbstractAction() {
			@Override public void actionPerformed(java.awt.event.ActionEvent e) { toggleFullscreen(); }
		});
        this.setVisible(true);
	}
	
	public void render(MazeData data) {
		this.data = data;
		repaint();
	}

	public void setControlListener(ControlListener listener) {
		this.controlListener = listener;
	}

	public void setSelectedMaze(String filename) {
		if (filename != null && mazeBox != null) mazeBox.setSelectedItem(filename);
	}

	public void setControlsEnabled(boolean enabled) {
		if (algorithmBox != null) algorithmBox.setEnabled(enabled);
		if (runButton != null) runButton.setEnabled(enabled);
		if (resetButton != null) resetButton.setEnabled(enabled);
		if (fullscreenButton != null) fullscreenButton.setEnabled(enabled);
	}

	private JPanel buildControlPanel() {
		JPanel panel = new JPanel();
		panel.add(new JLabel("Maze: "));
		if (this.mazeFiles != null && this.mazeFiles.length > 0) mazeBox = new JComboBox<>(this.mazeFiles); else mazeBox = new JComboBox<>(new String[] { "m100_100.txt" });
		panel.add(mazeBox);
		panel.add(new JLabel("Algorithm:"));
		this.algorithmBox = new JComboBox<>(new String[]{
			"Dijkstra", "A*", "BFS", "Genetic"
		});
		panel.add(algorithmBox);
		runButton = new JButton("Run");
		runButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (controlListener != null) {
					String name = (String) algorithmBox.getSelectedItem();
					controlListener.onRunRequested(name);
				}
			}
		});
		panel.add(runButton);
		resetButton = new JButton("Reset");
		resetButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (controlListener != null) {
					controlListener.onResetRequested();
				}
			}
		});
		panel.add(resetButton);
		fullscreenButton = new JButton("Fullscreen");
		fullscreenButton.addActionListener(new ActionListener() {
			@Override public void actionPerformed(ActionEvent e) { toggleFullscreen(); }
		});
		panel.add(fullscreenButton);

		panel.add(new JLabel("Speed:"));
		speedSlider = new JSlider(0, 200, 30); // delay ms
		speedSlider.setMajorTickSpacing(50);
		speedSlider.setMinorTickSpacing(10);
		speedSlider.setPaintTicks(true);
		panel.add(speedSlider);

		// Metrics display
		costLabel = new JLabel("Cost: -");
		stepsLabel = new JLabel("Steps: -");
		visitedLabel = new JLabel("Visited: -");
		timeLabel = new JLabel("Time: -ms");
		visitedWeightLabel = new JLabel("Visited Weight: -");
		panel.add(costLabel);
		panel.add(stepsLabel);
		panel.add(visitedLabel);
		panel.add(timeLabel);
		panel.add(visitedWeightLabel);
		// Maze selector change event
		if (mazeBox != null) {
			mazeBox.addActionListener(new ActionListener() {
				@Override public void actionPerformed(ActionEvent e) {
					if (controlListener != null) controlListener.onMazeSelected((String) mazeBox.getSelectedItem());
				}
			});
		}
		return panel;
	}

	// Toggle fullscreen for the frame using simple undecorated maximized state
	public void toggleFullscreen() {
		if (!fullscreen) {
			windowedBounds = this.getBounds();
			this.dispose();
			this.setUndecorated(true);
			this.setVisible(true);
			this.setExtendedState(JFrame.MAXIMIZED_BOTH);
			fullscreen = true;
		} else {
			this.dispose();
			this.setUndecorated(false);
			if (windowedBounds != null) this.setBounds(windowedBounds);
			this.setVisible(true);
			fullscreen = false;
		}
	}
	
	public void paint(MazeUtil util) {
		int cw = Math.max(1, canvas.getWidth());
		int ch = Math.max(1, canvas.getHeight());
		int w = Math.max(1, cw / data.M());
		int h = Math.max(1, ch / data.N());
		for(int i = 0; i < data.N(); i++) {
			for(int j = 0; j < data.M(); j++) {
				if(data.getMazeChar(i, j) == MazeData.WALL) {
					util.setColor(MazeUtil.LightBlue);
				}else {
					util.setColor(MazeUtil.White);
				}
				if(data.path[i][j]) {
					util.setColor(MazeUtil.Yellow);
				}
				if(data.result[i][j]) {
					util.setColor(MazeUtil.Red);
				}
				util.fillRectangle(j * w, i * h, w, h);

				// Draw Start/Goal labels or weight for road cells
				if (i == data.getEntranceX() && j == data.getEntranceY()) {
					util.setColor(Color.BLACK);
					util.drawCenteredString("S", j * w, i * h, w, h);
				} else if (i == data.getExitX() && j == data.getExitY()) {
					util.setColor(Color.BLACK);
					util.drawCenteredString("G", j * w, i * h, w, h);
				} else if (data.getMazeChar(i, j) == MazeData.ROAD && data.weight != null && data.weight[i][j] > 0) {
					util.setColor(Color.BLACK);
					util.drawCenteredString(Integer.toString(data.weight[i][j]), j * w, i * h, w, h);
				}
			}
		}
	}
	
	private class MazeCanvas extends JPanel{

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			MazeUtil util = MazeUtil.getInstance(g);
			if(data != null) {
				MazeFrame.this.paint(util);
			}
		}

		@Override
		public Dimension getPreferredSize() {
			return new Dimension(canvasWidth, canvasHeight);
		}
		
		
	}

	public static interface ControlListener {
		void onRunRequested(String algorithmName);
		void onResetRequested();
		void onMazeSelected(String mazeFile);
	}

	public void setCanvasSize(int w, int h) {
		this.canvasWidth = w;
		this.canvasHeight = h;
		if (this.canvas != null) {
			this.canvas.setPreferredSize(new Dimension(w, h));
		}
		this.pack();
	}

	public int getDelayMs() {
		return speedSlider != null ? speedSlider.getValue() : 10;
	}

	public void updateMetrics(Integer cost, Integer steps, Integer visited, Long timeMs, String algoName, Integer visitedWeightSum) {
		if (algoName != null) {
			setTitle("Maze Solver - " + algoName);
		}
		if (costLabel != null) costLabel.setText("Cost: " + (cost != null ? cost : "-"));
		if (stepsLabel != null) stepsLabel.setText("Steps: " + (steps != null ? steps : "-"));
		if (visitedLabel != null) visitedLabel.setText("Visited: " + (visited != null ? visited : "-"));
		if (timeLabel != null) timeLabel.setText("Time: " + (timeMs != null ? timeMs : "-") + "ms");
		if (visitedWeightLabel != null) visitedWeightLabel.setText("Visited Weight: " + (visitedWeightSum != null ? visitedWeightSum : "-"));
	}
}
