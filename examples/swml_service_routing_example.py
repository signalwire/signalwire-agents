#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Custom Routing Callbacks with SWMLService

This example demonstrates how to implement dynamic request routing using the
register_routing_callback method in SWMLService. It shows how to:

1. Create multiple endpoint paths (/main, /customer, /product) with a single service
2. Register callback functions to process requests at specific paths
3. Create and serve different SWML content based on the request path
4. Examine request data to make routing decisions
5. Use the on_request method with callback_path to serve path-specific content

This functionality allows creating sophisticated multi-purpose services that can
handle different types of requests at different endpoints, all from a single
SWMLService instance.
"""

import json
import os
import sys
import argparse
from typing import Dict, Optional, Any
from fastapi import Request

from signalwire_agents.core.swml_service import SWMLService


class RoutingExample(SWMLService):
    """
    Example SWMLService with custom routing callbacks
    """
    
    def __init__(self):
        # Initialize the SWMLService with basic configuration
        super().__init__(
            name="routing-example",
            route="/main",
            host="0.0.0.0",
            port=3000
        )
        
        # Create a basic SWML document
        self.reset_document()
        self.add_verb("answer", {})
        self.add_verb("play", {"url": "say:Hello from the main service!"})
        self.add_verb("hangup", {})
        
        # Register a customer callback route
        self.register_customer_route()
        
        # Register a product callback route
        self.register_product_route()
        
    def register_customer_route(self):
        """
        Register a callback for customer-related requests
        """
        def customer_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            """
            Callback for customer-related requests
            """
            print(f"Received customer request with body: {json.dumps(body, indent=2)}")
            
            # If the request contains a specific customer_id, redirect to a "virtual" endpoint
            if "customer_id" in body:
                customer_id = body["customer_id"]
                print(f"Routing to virtual customer endpoint for ID: {customer_id}")
                
                # In a real implementation, you might redirect to another service
                # For this example, we'll just handle it here
                return None
            
            # Otherwise, continue with normal processing - return None to not redirect
            return None
            
        # Register the callback at the /customer path
        self.register_routing_callback(customer_callback, path="/customer")
        
        # Create a custom SWML document for the customer path
        customer_section = "customer_section"
        self.add_section(customer_section)
        self.add_verb_to_section(customer_section, "answer", {})
        self.add_verb_to_section(customer_section, "play", {"url": "say:Hello from the customer service!"})
        self.add_verb_to_section(customer_section, "hangup", {})
    
    def register_product_route(self):
        """
        Register a callback for product-related requests
        """
        def product_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            """
            Callback for product-related requests
            """
            print(f"Received product request with body: {json.dumps(body, indent=2)}")
            
            # If the request contains a specific product_id, redirect to a "virtual" endpoint
            if "product_id" in body:
                product_id = body["product_id"]
                print(f"Routing to virtual product endpoint for ID: {product_id}")
                
                # In a real implementation, you might redirect to another service
                # For this example, we'll just handle it here
                return None
            
            # Otherwise, continue with normal processing - return None to not redirect
            return None
            
        # Register the callback at the /product path
        self.register_routing_callback(product_callback, path="/product")
        
        # Create a custom SWML document for the product path
        product_section = "product_section"
        self.add_section(product_section)
        self.add_verb_to_section(product_section, "answer", {})
        self.add_verb_to_section(product_section, "play", {"url": "say:Hello from the product service!"})
        self.add_verb_to_section(product_section, "hangup", {})
    
    def on_request(self, request_data: Optional[dict] = None, callback_path: Optional[str] = None) -> Optional[dict]:
        """
        Called when SWML is requested, with request data when available
        
        Args:
            request_data: Optional dictionary containing the parsed POST body
            callback_path: Optional callback path from request
            
        Returns:
            Optional dict to modify/augment the SWML document
        """
        # Now we can directly check the callback_path
        if callback_path == "/customer":
            print(f"Serving customer section based on callback_path: {callback_path}")
            return {
                "sections": {
                    "main": self.get_document()["sections"]["customer_section"]
                }
            }
        
        if callback_path == "/product":
            print(f"Serving product section based on callback_path: {callback_path}")
            return {
                "sections": {
                    "main": self.get_document()["sections"]["product_section"]
                }
            }
        
        # If we don't have a callback_path, check the request_data
        if request_data:
            # Check for customer-specific data
            if "customer_id" in request_data or "customer_section" in request_data:
                print(f"Serving customer section based on request data")
                return {
                    "sections": {
                        "main": self.get_document()["sections"]["customer_section"]
                    }
                }
            
            # Check for product-specific data
            if "product_id" in request_data or "product_section" in request_data:
                print(f"Serving product section based on request data")
                return {
                    "sections": {
                        "main": self.get_document()["sections"]["product_section"]
                    }
                }
        
        # Log which section we're serving
        print(f"Serving default main section")
        
        # Otherwise, return the default document
        return None


def main():
    # Create an instance of our example service
    service = RoutingExample()
    return service


if __name__ == "__main__":
    service = main()
    # Start the service
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    service.serve() 