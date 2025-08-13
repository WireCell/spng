#include "WireCellSpng/TarStreamer.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " archive.tar\n";
        return 2;
    }

    try {
        TarStreamer ts(argv[1]);
        std::cout << "Total entries: " << ts.total_files() << "\n";

        // Stream everything: print names; if JSON/NPY, parse/peek first 10 numbers.
        while (ts.next()) {
            const auto& e = ts.current();
            std::cout << "Entry: " << e.name << " (" << e.size << " bytes)\n";

            if (e.name.size() >= 5 && e.name.rfind(".json") == e.name.size() - 5) {
                auto j = ts.read_current_json();
                std::cout << "  JSON keys: ";
                bool first = true;
                for (auto it = j.begin(); it != j.end(); ++it) {
                    std::cout << (first ? "" : ", ") << it.key();
                    first = false;
                }
                std::cout << "\n";
            }
            else if (e.name.size() >= 4 && e.name.rfind(".npy") == e.name.size() - 4) {
                auto arr = ts.read_current_npy();
                std::cout << "  NPY shape: [";
                for (size_t i = 0; i < arr.shape.size(); ++i) {
                    std::cout << arr.shape[i] << (i+1<arr.shape.size()? ", ":"");
                }
                std::cout << "]\n  First 10: ";
                size_t n = std::min<size_t>(10, arr.data.size());
                for (size_t i = 0; i < n; ++i) std::cout << arr.data[i] << (i+1<n? " ":"");
                std::cout << "\n";
            }
            else {
                // Not JSON/NPY: skip data fast
                ts.skip_current_data();
            }

            std::cout << "  Progress: " << ts.files_read()
                      << "/" << ts.total_files()
                      << " (remaining " << ts.files_remaining() << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}