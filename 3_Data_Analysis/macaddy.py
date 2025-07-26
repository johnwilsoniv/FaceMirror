import os
import random
import subprocess # Recommended for running shell commands and capturing output

def get_hex():
    """Returns a random hexadecimal character."""
    return random.choice("abcdef0123456789")

def generate_random_mac():
    """Generates a random MAC address in the format XX:XX:XX:XX:XX:XX."""
    new_mac = ""
    for _ in range(5): # Use '_' if you don't need the loop variable
        new_mac += get_hex() + get_hex() + ":"
    new_mac += get_hex() + get_hex()
    return new_mac

def get_current_mac(interface="en0"): # 'en0' is more common for Wi-Fi on modern Macs
    """
    Gets the current MAC address of the specified interface.
    Returns None if the MAC address cannot be retrieved.
    """
    try:
        # Using subprocess.check_output to capture the command's output
        # `ifconfig` is being deprecated; `networksetup` is preferred for MAC changes
        # However, `ifconfig` can still show the MAC.
        # `grep -oE` is good for extracting the MAC.
        output = subprocess.check_output(f"ifconfig {interface} | grep ether | grep -oE '[0-9a-fA-F:]{17}'", shell=True, text=True)
        return output.strip()
    except subprocess.CalledProcessError:
        return None

def change_mac_address(interface, new_mac):
    """
    Changes the MAC address of the specified interface using networksetup.
    Returns True on success, False on failure.
    """
    print(f"Attempting to change MAC address of {interface} to {new_mac}...")
    try:
        # macOS uses `networksetup` for changing MAC addresses.
        # You might need to disable and re-enable the interface for the change to take effect.
        # This is a common method, but sometimes a reboot might be required for persistence.

        # First, deactivate the network service for the interface (e.g., Wi-Fi)
        # This part is a bit tricky as `networksetup` operates on "services" not interfaces directly.
        # We need to find the service name corresponding to the interface.
        # For Wi-Fi, it's usually "Wi-Fi".
        # For Ethernet, it might be "Ethernet".
        # Let's try to assume "Wi-Fi" for `en0` and "Ethernet" for `en1` (or whatever the user provides)

        service_name = ""
        if interface == "en0":
            service_name = "Wi-Fi"
        elif interface == "en1":
            service_name = "Ethernet" # Or whatever your 'en1' interface is called in Network Preferences

        if not service_name:
            print(f"Error: Could not determine network service name for interface {interface}. Please specify manually if needed.")
            return False

        # Attempt to set the MAC address
        # Note: `networksetup` for changing MAC addresses is not straightforward
        # The common approach for changing MAC on macOS often involves a different method
        # or it's simply not officially supported in the same way as Linux `macchanger`.

        # A more common, though less persistent, way to set a MAC in macOS often involves
        # loading and unloading the interface, but `networksetup` is the *intended* way.
        # However, `networksetup -setmacaddress` requires the MAC address to be *permanent*.
        # For temporary changes, many resort to `ifconfig` tricks.

        # Let's stick with the `ifconfig` approach for simplicity as that's what you started with,
        # but acknowledge its limitations. It will change it temporarily until restart or network change.
        subprocess.check_call(f"sudo ifconfig {interface} ether {new_mac}", shell=True)
        print(f"Successfully sent command to change MAC address for {interface}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error changing MAC address: {e}")
        print("Please ensure you have administrator privileges (sudo) and the interface name is correct.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Define the interface you want to change. 'en0' is typically Wi-Fi on MacBooks.
    # You can find your interface name using `ifconfig` or `networksetup -listallhardwareports`.
    target_interface = "en0" # Or "en1" if you have an external adapter or older Mac

    print("--- MAC Address Changer for macOS ---")

    # Get and print the old MAC address
    old_mac = get_current_mac(target_interface)
    if old_mac:
        print(f"Old MAC address of {target_interface}: {old_mac}")
    else:
        print(f"Could not retrieve old MAC address for {target_interface}.")
        print("Please ensure the interface name is correct and it's active.")

    # Generate a new random MAC address
    new_random_mac = generate_random_mac()
    print(f"Generated new random MAC address: {new_random_mac}")

    # Attempt to change the MAC address
    # You will be prompted for your password due to `sudo`
    if change_mac_address(target_interface, new_random_mac):
        print("\nVerifying new MAC address...")
        # Give it a moment to apply, though `ifconfig` changes are usually immediate
        current_mac_after_change = get_current_mac(target_interface)
        if current_mac_after_change:
            print(f"New MAC address of {target_interface}: {current_mac_after_change}")
            if current_mac_after_change.lower() == new_random_mac.lower():
                print("MAC address successfully changed (at least temporarily).")
            else:
                print("MAC address shown by ifconfig does not match the requested MAC address.")
                print("The change might not have been applied or reflected immediately.")
        else:
            print(f"Could not retrieve MAC address for {target_interface} after attempting change.")
    else:
        print("Failed to change MAC address.")