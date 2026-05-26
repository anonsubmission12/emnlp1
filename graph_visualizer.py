import json
import os
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Optional


class GraphVisualizer:
    """
    Real-time graph visualization for intent and strategy graphs.
    Creates live-updating HTML that refreshes automatically during generation.
    """
    
    def __init__(self, output_dir="output/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.snapshots = []
        self.snapshots_file = f"{output_dir}/snapshots.json"
        self.server_thread = None
        self.server = None
        
        # Initialize empty snapshots file
        self._save_snapshots()
        
        # Generate initial HTML
        self._generate_html()
    
    def start_server(self, port=8000):
        """Start HTTP server for live visualization"""
        output_dir = self.output_dir  # Capture in closure
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=output_dir, **kwargs)
            
            def log_message(self, format, *args):
                pass  # Suppress server logs
        
        try:
            self.server = HTTPServer(('localhost', port), Handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"\n  [Live] Live visualization at: http://localhost:{port}/graph_evolution.html")
            print(f"  [Info] Open this URL in your browser to see real-time updates!\n")
        except Exception as e:
            print(f"  [Error] Could not start server: {e}")
    
    def capture_snapshot(self, intent_graph, strategy_graph, conversation_id, turn_no, update_info=None):
        """Capture current graph state and update live visualization"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "turn_no": turn_no,
            "intent_graph": intent_graph.to_dict(),
            "strategy_graph": strategy_graph.to_dict(),
            "update_info": update_info
        }
        self.snapshots.append(snapshot)
        
        # Save snapshots to JSON file (for live reload)
        self._save_snapshots()
    
    def _save_snapshots(self):
        """Save snapshots to JSON file"""
        with open(self.snapshots_file, "w", encoding="utf-8") as f:
            json.dump(self.snapshots, f, indent=2)
    
    def _generate_html(self):
        """Generate live-updating HTML visualization"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Graph Evolution - Live View</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #fff; }
        h1 { text-align: center; color: #00ff88; margin-bottom: 10px; }
        .live-indicator { text-align: center; color: #00ff88; font-size: 14px; margin-bottom: 20px; }
        .live-dot { display: inline-block; width: 10px; height: 10px; background: #00ff88; border-radius: 50%; margin-right: 8px; animation: blink 1.5s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        .graph-container { display: flex; gap: 40px; margin: 20px 0; justify-content: center; }
        .graph { flex: 1; max-width: 600px; }
        .graph h2 { text-align: center; color: #00ccff; }
        svg { background: #1a1a1a; border-radius: 12px; border: 2px solid #333; box-shadow: 0 4px 20px rgba(0,255,136,0.2); }
        .node circle { stroke: #fff; stroke-width: 2.5px; transition: all 0.3s; }
        .node:hover circle { r: 25; stroke: #00ff88; stroke-width: 3px; }
        .node text { pointer-events: none; font-weight: 600; }
        .link { stroke-opacity: 0.7; transition: all 0.3s; }
        .updated { stroke: #00ff00 !important; stroke-width: 4px !important; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .info { background: #1a1a1a; padding: 20px; border-radius: 12px; margin: 10px 0; border: 2px solid #333; }
        .info-row { display: flex; justify-content: space-between; margin: 8px 0; }
        .info-label { color: #888; }
        .info-value { color: #00ff88; font-weight: 600; }
        #update-info { margin-top: 10px; padding: 10px; background: #0d4d3d; border-left: 4px solid #00ff88; border-radius: 4px; }
        .legend { display: flex; justify-content: center; gap: 30px; margin: 20px 0; }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-line { width: 40px; height: 4px; border-radius: 2px; }
    </style>
</head>
<body>
    <h1> Graph Evolution - Live View</h1>
    <div class="live-indicator">
        <span class="live-dot"></span>
        <span>Auto-refreshing every 2 seconds</span>
    </div>
    
    <div class="info">
        <div class="info-row">
            <span class="info-label">Total Snapshots:</span>
            <span class="info-value" id="total-snapshots">0</span>
        </div>
        <div class="info-row">
            <span class="info-label">Latest Conversation:</span>
            <span class="info-value" id="conv-id">-</span>
        </div>
        <div class="info-row">
            <span class="info-label">Latest Turn:</span>
            <span class="info-value" id="turn-num">-</span>
        </div>
        <div class="info-row">
            <span class="info-label">Last Update:</span>
            <span class="info-value" id="last-update">-</span>
        </div>
        <div id="update-info" style="display:none;"></div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-line" style="background: #00ff00;"></div>
            <span>High Weight (0.20-0.25)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: #ffff00;"></div>
            <span>Medium Weight (0.16-0.20)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: #ff6600;"></div>
            <span>Low Weight (0.08-0.12)</span>
        </div>
    </div>
    
    <div class="graph-container">
        <div class="graph">
            <h2>Intent Graph</h2>
            <svg id="intent-graph" width="600" height="500"></svg>
        </div>
        <div class="graph">
            <h2>Strategy Graph</h2>
            <svg id="strategy-graph" width="600" height="500"></svg>
        </div>
    </div>
    
    <script>
        let snapshots = [];
        let lastSnapshotCount = 0;
        
        function renderGraph(graphData, svgId) {
            const svg = d3.select(`#${svgId}`);
            svg.selectAll("*").remove();
            
            const width = 600, height = 500;
            
            const nodes = Object.keys(graphData.weights).map(node => ({id: node}));
            const links = [];
            
            Object.entries(graphData.weights).forEach(([source, targets]) => {
                Object.entries(targets).forEach(([target, weight]) => {
                    links.push({source, target, weight});
                });
            });
            
            // More sensitive color scale for small weights (0.1-0.25 range)
            const colorScale = d3.scaleLinear()
                .domain([0.08, 0.12, 0.16, 0.20, 0.25])
                .range(['#ff0000', '#ff6600', '#ffaa00', '#ffff00', '#00ff00'])
                .clamp(true);
            
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(120))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(30));
            
            // Draw links
            const linkGroup = svg.append("g");
            
            const link = linkGroup
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke", d => colorScale(d.weight))
                .attr("stroke-width", d => Math.max(2, d.weight * 30));  // More visible
            
            // Add weight labels on edges
            const linkLabels = linkGroup
                .selectAll("text")
                .data(links)
                .enter().append("text")
                .attr("class", "link-label")
                .attr("text-anchor", "middle")
                .attr("fill", "#fff")
                .attr("font-size", "10px")
                .attr("font-weight", "bold")
                .style("pointer-events", "none")
                .text(d => d.weight.toFixed(3));  // Show weight value
            
            // Draw nodes
            const node = svg.append("g")
                .selectAll("g")
                .data(nodes)
                .enter().append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            node.append("circle")
                .attr("r", 20)
                .attr("fill", "#00ccff");
            
            node.append("text")
                .text(d => d.id)
                .attr("text-anchor", "middle")
                .attr("dy", 4)
                .attr("fill", "#000")
                .style("font-size", "11px")
                .style("font-weight", "600");
            
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                // Position labels at midpoint of edges
                linkLabels
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2);
                
                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        function updateVisualization() {
            fetch('snapshots.json')
                .then(response => response.json())
                .then(data => {
                    snapshots = data;
                    
                    if (snapshots.length > 0) {
                        const latest = snapshots[snapshots.length - 1];
                        
                        document.getElementById('total-snapshots').textContent = snapshots.length;
                        document.getElementById('conv-id').textContent = latest.conversation_id;
                        document.getElementById('turn-num').textContent = latest.turn_no;
                        document.getElementById('last-update').textContent = new Date(latest.timestamp).toLocaleTimeString();
                        
                        const updateInfoDiv = document.getElementById('update-info');
                        if (latest.update_info) {
                            updateInfoDiv.style.display = 'block';
                            updateInfoDiv.innerHTML = `<strong> Latest Update:</strong> ${latest.update_info}`;
                        } else {
                            updateInfoDiv.style.display = 'none';
                        }
                        
                        // Only re-render if new snapshots were added
                        if (snapshots.length !== lastSnapshotCount) {
                            renderGraph(latest.intent_graph, 'intent-graph');
                            renderGraph(latest.strategy_graph, 'strategy-graph');
                            lastSnapshotCount = snapshots.length;
                        }
                    }
                })
                .catch(err => console.error('Error loading snapshots:', err));
        }
        
        // Initial load
        updateVisualization();
        
        // Auto-refresh every 2 seconds
        setInterval(updateVisualization, 2000);
    </script>
</body>
</html>"""
        
        with open(f"{self.output_dir}/graph_evolution.html", "w", encoding="utf-8") as f:
            f.write(html)
