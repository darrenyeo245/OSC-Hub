from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

print("Reward input gestartet")
print("Kommandos:")
print("  -1.0...1.0 -> numerischer Reward")
print("  r                   -> Episode manuell resetten")
print("  e                   -> Episode manuell beenden")
print("  q                   -> Ende")

while True:
    key = input("reward> ").strip()
    if key == "r":
        client.send_message("/episode/reset_manual", 1)
        print("Sent event: manual reset")
    elif key == "e":
        client.send_message("/episode/end", 1)
        print("Sent event: episode end")
    elif key.lower() == "q":
        client.send_message("/training/stop", 1)
        print("Sent event: training stop")
    elif key.lower() == "qs":
        client.send_message("/training/stop_save", 1)
        break
    else:
        try:
            value = float(key.replace(",", "."))
            if -1.0 <= value <= 1.0:
                client.send_message("/reward", value)
                print(f"Sent reward: {value}")
            else:
                print("Reward muss zwischen -1.0 und 1.0 liegen")
        except ValueError:
            print("Unbekanntes Kommando")

