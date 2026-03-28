#!/bin/bash
#
# TradeStation Integration Test Setup and Runner
#
# This script helps you configure credentials and run integration tests safely.
#

echo "=========================================================================="
echo "TradeStation Integration Test Setup"
echo "=========================================================================="
echo ""

# Check if credentials are already set
if [ -n "$TRADESTATION_SIM_CLIENT_ID" ] && [ -n "$TRADESTATION_SIM_CLIENT_SECRET" ]; then
    echo "✅ Credentials already configured!"
    echo ""
    echo "Ready to run tests. Choose an option:"
    echo ""
    echo "1) Run authentication tests (safe, no orders)"
    echo "2) Run market data tests (read-only)"
    echo "3) Run order lifecycle tests (SIM orders)"
    echo "4) Run all tests"
    echo "5) Exit"
    echo ""
    read -p "Choose option [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Running authentication tests..."
            /root/Silver-Bullet-ML-BMAD/.venv/bin/python /root/Silver-Bullet-ML-BMAD/run_integration_tests.py --module test_auth_flow
            ;;
        2)
            echo ""
            echo "Running market data tests..."
            /root/Silver-Bullet-ML-BMAD/.venv/bin/python /root/Silver-Bullet-ML-BMAD/run_integration_tests.py --module test_market_data_flow
            ;;
        3)
            echo ""
            echo "Running order lifecycle tests..."
            echo "⚠️  WARNING: This will place real SIM orders (no real money)"
            read -p "Continue? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                /root/Silver-Bullet-ML-BMAD/.venv/bin/python /root/Silver-Bullet-ML-BMAD/run_integration_tests.py --module test_order_flow
            else
                echo "Cancelled."
            fi
            ;;
        4)
            echo ""
            echo "Running all tests..."
            /root/Silver-Bullet-ML-BMAD/.venv/bin/python /root/Silver-Bullet-ML-BMAD/run_integration_tests.py
            ;;
        5)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option."
            exit 1
            ;;
    esac
else
    echo "Credentials not configured. Let's set them up!"
    echo ""
    echo "You'll need your TradeStation SIM API credentials:"
    echo "  1. Log into TradeStation Developer Portal"
    echo "  2. Create a SIM application"
    echo "  3. Copy your Client ID and Client Secret"
    echo ""
    echo "How would you like to provide credentials?"
    echo ""
    echo "1) Enter now (temporary, current session only)"
    echo "2) Save to ~/.bashrc (permanent, persists across sessions)"
    echo "3) Cancel"
    echo ""
    read -p "Choose option [1-3]: " setup_choice

    case $setup_choice in
        1)
            echo ""
            echo "Enter your TradeStation SIM credentials:"
            read -p "Client ID: " client_id
            read -p "Client Secret: " client_secret

            if [ -z "$client_id" ] || [ -z "$client_secret" ]; then
                echo "❌ Credentials cannot be empty."
                exit 1
            fi

            export TRADESTATION_SIM_CLIENT_ID="$client_id"
            export TRADESTATION_SIM_CLIENT_SECRET="$client_secret"

            echo ""
            echo "✅ Credentials set for this session!"
            echo ""
            echo "You can now run tests with:"
            echo "  python run_integration_tests.py"
            ;;
        2)
            echo ""
            echo "Enter your TradeStation SIM credentials:"
            read -p "Client ID: " client_id
            read -p "Client Secret: " client_secret

            if [ -z "$client_id" ] || [ -z "$client_secret" ]; then
                echo "❌ Credentials cannot be empty."
                exit 1
            fi

            echo "export TRADESTATION_SIM_CLIENT_ID=\"$client_id\"" >> ~/.bashrc
            echo "export TRADESTATION_SIM_CLIENT_SECRET=\"$client_secret\"" >> ~/.bashrc

            echo ""
            echo "✅ Credentials saved to ~/.bashrc"
            echo ""
            echo "Run 'source ~/.bashrc' or restart your shell to use them."
            ;;
        3)
            echo "Cancelled. Credentials not configured."
            echo ""
            echo "To manually set credentials later, run:"
            echo "  export TRADESTATION_SIM_CLIENT_ID='your_client_id'"
            echo "  export TRADESTATION_SIM_CLIENT_SECRET='your_client_secret'"
            exit 0
            ;;
        *)
            echo "Invalid option."
            exit 1
            ;;
    esac
fi

echo ""
echo "=========================================================================="
echo "Done!"
echo "=========================================================================="
