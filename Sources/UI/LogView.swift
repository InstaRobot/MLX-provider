import SwiftUI

struct LogView: View {
    @State private var logs: [LogEntry] = []
    @State private var isAutoScroll = true
    
    var body: some View {
        GroupBox("Activity Log") {
            VStack(spacing: 0) {
                // Log entries
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(logs) { entry in
                                LogEntryView(entry: entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(8)
                    }
                    .background(Color(nsColor: .textBackgroundColor))
                    .onChange(of: logs.count) { _, _ in
                        if isAutoScroll, let lastId = logs.last?.id {
                            withAnimation {
                                proxy.scrollTo(lastId, anchor: .bottom)
                            }
                        }
                    }
                }
                
                Divider()
                
                // Controls
                HStack {
                    Toggle("Auto-scroll", isOn: $isAutoScroll)
                        .toggleStyle(.checkbox)
                        .font(.caption)
                    
                    Spacer()
                    
                    Button(action: clearLogs) {
                        Label("Clear", systemImage: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
            }
        }
        .frame(minHeight: 150, idealHeight: 200, maxHeight: 300)
    }
    
    private func clearLogs() {
        logs.removeAll()
    }
}

// MARK: - Log Entry

struct LogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let level: LogLevel
    let message: String
    
    enum LogLevel {
        case info
        case warning
        case error
        case debug
        
        var color: Color {
            switch self {
            case .info: return .primary
            case .warning: return .orange
            case .error: return .red
            case .debug: return .secondary
            }
        }
        
        var icon: String {
            switch self {
            case .info: return "info.circle"
            case .warning: return "exclamationmark.triangle"
            case .error: return "xmark.circle"
            case .debug: return "ant"
            }
        }
    }
}

// MARK: - Log Entry View

struct LogEntryView: View {
    let entry: LogEntry
    
    private var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: entry.timestamp)
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(timeString)
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.secondary)
            
            Image(systemName: entry.level.icon)
                .font(.caption2)
                .foregroundColor(entry.level.color)
            
            Text(entry.message)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(entry.level.color)
                .textSelection(.enabled)
            
            Spacer()
        }
    }
}

// MARK: - Log Manager

@MainActor
final class LogManager: ObservableObject {
    static let shared = LogManager()
    
    @Published var entries: [LogEntry] = []
    
    private let maxEntries = 500
    
    func log(_ message: String, level: LogEntry.LogLevel = .info) {
        let entry = LogEntry(timestamp: Date(), level: level, message: message)
        entries.append(entry)
        
        // Trim old entries
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }
    
    func info(_ message: String) { log(message, level: .info) }
    func warning(_ message: String) { log(message, level: .warning) }
    func error(_ message: String) { log(message, level: .error) }
    func debug(_ message: String) { log(message, level: .debug) }
    
    func clear() {
        entries.removeAll()
    }
}

// MARK: - Preview

#Preview {
    LogView()
        .frame(height: 300)
        .padding()
}
