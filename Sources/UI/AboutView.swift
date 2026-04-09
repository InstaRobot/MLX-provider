import SwiftUI

struct AboutView: View {
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack(spacing: 24) {
            // App Icon
            Image(systemName: "cpu.fill")
                .font(.system(size: 80))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .padding(.top, 20)
            
            // App Name & Version
            VStack(spacing: 4) {
                Text("MLX-provider")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("Version 1.0.0")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            // Description
            Text("OpenAI-compatible REST API server for MLX models on Apple Silicon")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 32)
            
            Divider()
                .padding(.horizontal, 40)
            
            // Requirements
            VStack(alignment: .leading, spacing: 12) {
                Text("Requirements")
                    .font(.headline)
                
                RequirementRow(icon: "cpu.fill", text: "Apple Silicon Mac (M1/M2/M3/M4)")
                RequirementRow(icon: "desktopcomputer", text: "macOS 15.0+")
                RequirementRow(icon: "function", text: "MLX Framework")
                RequirementRow(icon: "swift", text: "Xcode 16+")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 40)
            
            // Supported Models
            VStack(alignment: .leading, spacing: 12) {
                Text("Supported Models")
                    .font(.headline)
                
                FlowLayout(spacing: 8) {
                    ModelTag(text: "Llama")
                    ModelTag(text: "Mistral")
                    ModelTag(text: "Qwen")
                    ModelTag(text: "Gemma")
                    ModelTag(text: "Phi")
                    ModelTag(text: "OpenELM")
                    ModelTag(text: "+1000 more")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 40)
            
            Spacer()
            
            // Links
            HStack(spacing: 24) {
                Link(destination: URL(string: "https://github.com/ml-explore/mlx")!) {
                    Label("MLX", systemImage: "link")
                }
                
                Link(destination: URL(string: "https://github.com/ml-explore/mlx-swift-lm")!) {
                    Label("mlx-swift-lm", systemImage: "link")
                }
                
                Link(destination: URL(string: "https://huggingface.co/mlx-community")!) {
                    Label("HuggingFace", systemImage: "cloud")
                }
            }
            .font(.caption)
            
            // Copyright
            Text("Copyright © 2024 InstaRobot")
                .font(.caption2)
                .foregroundColor(.secondary)
                .padding(.bottom, 20)
        }
        .frame(width: 500, height: 520)
    }
}

// MARK: - Requirement Row

struct RequirementRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 20)
                .foregroundColor(.accentColor)
            
            Text(text)
                .font(.subheadline)
        }
    }
}

// MARK: - Model Tag

struct ModelTag: View {
    let text: String
    
    var body: some View {
        Text(text)
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Color.accentColor.opacity(0.1))
            .foregroundColor(.accentColor)
            .cornerRadius(12)
    }
}

// MARK: - Flow Layout

struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = FlowResult(
            in: proposal.replacingUnspecifiedDimensions().width,
            subviews: subviews,
            spacing: spacing
        )
        return result.size
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = FlowResult(
            in: bounds.width,
            subviews: subviews,
            spacing: spacing
        )
        
        for (index, subview) in subviews.enumerated() {
            let point = result.points[index]
            subview.place(at: CGPoint(x: bounds.minX + point.x, y: bounds.minY + point.y), proposal: .unspecified)
        }
    }
    
    struct FlowResult {
        var size: CGSize = .zero
        var points: [CGPoint] = []
        
        init(in maxWidth: CGFloat, subviews: Subviews, spacing: CGFloat) {
            var currentX: CGFloat = 0
            var currentY: CGFloat = 0
            var lineHeight: CGFloat = 0
            
            for subview in subviews {
                let size = subview.sizeThatFits(.unspecified)
                
                if currentX + size.width > maxWidth && currentX > 0 {
                    currentX = 0
                    currentY += lineHeight + spacing
                    lineHeight = 0
                }
                
                points.append(CGPoint(x: currentX, y: currentY))
                lineHeight = max(lineHeight, size.height)
                currentX += size.width + spacing
                
                self.size.width = max(self.size.width, currentX - spacing)
                self.size.height = currentY + lineHeight
            }
        }
    }
}

// MARK: - Preview

#Preview {
    AboutView()
}
