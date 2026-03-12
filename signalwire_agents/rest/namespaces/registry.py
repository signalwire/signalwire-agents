"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
10DLC Campaign Registry namespace — brands, campaigns, orders, numbers.
"""

from .._base import BaseResource


class RegistryBrands(BaseResource):
    """10DLC brand management."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def get(self, brand_id):
        return self._http.get(self._path(brand_id))

    def list_campaigns(self, brand_id, **params):
        return self._http.get(self._path(brand_id, "campaigns"), params=params or None)

    def create_campaign(self, brand_id, **kwargs):
        return self._http.post(self._path(brand_id, "campaigns"), body=kwargs)


class RegistryCampaigns(BaseResource):
    """10DLC campaign management."""

    def get(self, campaign_id):
        return self._http.get(self._path(campaign_id))

    def update(self, campaign_id, **kwargs):
        return self._http.put(self._path(campaign_id), body=kwargs)

    def list_numbers(self, campaign_id, **params):
        return self._http.get(self._path(campaign_id, "numbers"), params=params or None)

    def list_orders(self, campaign_id, **params):
        return self._http.get(self._path(campaign_id, "orders"), params=params or None)

    def create_order(self, campaign_id, **kwargs):
        return self._http.post(self._path(campaign_id, "orders"), body=kwargs)


class RegistryOrders(BaseResource):
    """10DLC assignment order management."""

    def get(self, order_id):
        return self._http.get(self._path(order_id))


class RegistryNumbers(BaseResource):
    """10DLC number assignment management."""

    def delete(self, number_id):
        return self._http.delete(self._path(number_id))


class RegistryNamespace:
    """10DLC Campaign Registry namespace."""

    def __init__(self, http):
        base = "/api/relay/rest/registry/beta"
        self.brands = RegistryBrands(http, f"{base}/brands")
        self.campaigns = RegistryCampaigns(http, f"{base}/campaigns")
        self.orders = RegistryOrders(http, f"{base}/orders")
        self.numbers = RegistryNumbers(http, f"{base}/numbers")
